# VLLM源码阅读

# 目标

1. 了解vllm拉拉起过程
2. vllm核心创新点
3. 

# VLLM核心架构



整体框架结构如图所示

![vllm结构图](/Users/wangjingxin/knowlege_library/大模型相关知识/VLLM/VLLM源码阅读/images/image-20260328105755410-4666679.png)



如下是针对vllm的使用方法，进入模型是llm，也就是我们的入口文件

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

按照系统启动流程，依次深入以下核心组件：

## class LLM

会拉起LLMEngine，调用from_engine_args, 内部会初始化executor

```python
class LLM:
    def __init__(self,...) -> None:

				....
        engine_args = EngineArgs(...)

				...
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        ) 
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()

        self.supported_tasks = self.llm_engine.get_supported_tasks()
        self.chat_template_config = ChatTemplateConfig(chat_template=self.chat_template)
        self.pooling_io_processors = init_pooling_io_processors(
            supported_tasks=supported_tasks,
            model_config=self.model_config,
            renderer=self.renderer,
            chat_template_config=self.chat_template_config,
        )
    
   
```

```python
  def from_engine_args(
      cls,
      engine_args: EngineArgs,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      stat_loggers: list[StatLoggerFactory] | None = None,
      enable_multiprocessing: bool = False,
  ) -> "LLMEngine":
      """Creates an LLM engine from the engine arguments."""

      # Create the engine configs.
      vllm_config = engine_args.create_engine_config(usage_context)
      executor_class = Executor.get_class(vllm_config)

      if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
          logger.debug("Enabling multiprocessing for LLMEngine.")
          enable_multiprocessing = True

      # Create the LLMEngine.
      return cls(
          vllm_config=vllm_config,
          executor_class=executor_class,
          log_stats=not engine_args.disable_log_stats,
          usage_context=usage_context,
          stat_loggers=stat_loggers,
          multiprocess_mode=enable_multiprocessing,
      )
```



## class LLMEngine

```python
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(...) -> None:


        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None

        self.io_processor = get_io_processor(
            self.vllm_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # Don't keep the dummy data in memory
        self.reset_mm_cache()
```



## class EngineCoreClient

```python
class EngineCoreClient(ABC):
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)
    
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            return DPLBAsyncMPClient(*client_args)
        return AsyncMPClient(*client_args)
```

DPLBAsyncMPClient 一路初始化，会走完全部的父类，触发

```python
  with launch_core_engines(
      vllm_config, executor_class, log_stats, addresses
  ) as (engine_manager, coordinator, addresses, tensor_queue):
      self.resources.coordinator = coordinator
      self.resources.engine_manager = engine_manager
```



```python
def launch_core_engines(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
        Queue | None,
    ]
]:
  	...


    if multimodal_config is not None and multimodal_config.mm_tensor_ipc == "torch_shm":
        tensor_queue = get_mp_context().Queue()

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            vllm_config=vllm_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses, tensor_queue
        return

    if offline_mode:
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # Rank 0 holds Coordinator, so it handshakes with all Cores
        # in both external dplb and internal dplb mode.
        # Note this also covers the case where we have zero local engines
        # and rank 0 is headless.
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]


    handshake_local_only = offline_mode or local_engine_count == dp_size

    # NOTE(yongji): handling scaling from intra-node to inter-node
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port
    )

    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

        yield local_engine_manager, coordinator, addresses, tensor_queue

        # Now wait for engines to start.
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )
```

## class EngineCoreProc

父类为class EngineCore, 内部会初始化好kv cache的东西

1. 初始化scheduler
2. 准备kv cache

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
      ....

        # Setup Model.
        self.model_executor = executor_class(vllm_config)

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self._eep_scale_up_before_kv_init()

        # Setup KV Caches and update CacheConfig after profiling.
        kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
        
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = mm_registry.engine_receiver_cache_from_config(
            vllm_config
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        self.is_ec_consumer = (
            vllm_config.ec_transfer_config is None
            or vllm_config.ec_transfer_config.is_ec_consumer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )
...
		def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            decorate_logs()

            if data_parallel and vllm_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                vllm_config.kv_transfer_config.engine_id = (
                    f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
                )
                logger.debug(
                    "Setting kv_transfer_config.engine_id to %s",
                    vllm_config.kv_transfer_config.engine_id,
                )

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

            def wakeup_engine():
                # Wakes up idle engine via input_queue when shutdown is requested
                # Not safe in a signal handler - we may interrupt the main thread
                # while it is holding the non-reentrant input_queue.mutex
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum, frame):
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()
...
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()#不停放入block
            # 2) Step the engine core and return the outputs.
            self._process_engine_step() #Called only when there are unfinished local requests.
```

## dp分布在单节点情况下 EngineManager

CoreEngineProcManager，然后会启动dp个EngineCoreProc.run_engine_core进程

```python
class CoreEngineProcManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of background processes used by the AsyncLLM and LLMEngine.
    """

    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        context = get_mp_context()
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = []
        local_dp_ranks = []
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        self._finalizer = weakref.finalize(self, shutdown, self.processes)

        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # Adjust device control in DP for non-CUDA platforms
                # as well as external and ray launchers
                # For CUDA platforms, we use torch.accelerator.set_device_index()()
                if is_dp and (
                    not current_platform.is_cuda_alike()
                    or vllm_config.parallel_config.use_ray
                ):
                    with set_device_control_env_var(vllm_config, local_dp_rank):
                        proc.start()
                else:
                    proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.shutdown()
```



## dp多节点创建工作进程（Worker）

**基础概念：**

1. node_rank
2. nnodes_within_dp 
3. rank
4. tensor_parallel_size

以多节点为例

LLMEngine类中：

```python
class LLMEngine：
  def from_engine_args(
      cls,
      engine_args: EngineArgs,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      stat_loggers: list[StatLoggerFactory] | None = None,
      enable_multiprocessing: bool = False,
  ) -> "LLMEngine":
      """Creates an LLM engine from the engine arguments."""

      # Create the engine configs.
      vllm_config = engine_args.create_engine_config(usage_context)
      executor_class = Executor.get_class(vllm_config) # 初始化Executor，拿到真实executor类别进行初始化

      if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
          logger.debug("Enabling multiprocessing for LLMEngine.")
          enable_multiprocessing = True

      # Create the LLMEngine.
      return cls(
          vllm_config=vllm_config,
          executor_class=executor_class,
          log_stats=not engine_args.disable_log_stats,
          usage_context=usage_context,
          stat_loggers=stat_loggers,
          multiprocess_mode=enable_multiprocessing,
      )
```

`class MultiprocExecutor`在初始化时，会先实例化其基类 `class Executor`，随后执行自身的初始化方法 `_init_executor` 会拉起全部的Worker进程

```python
class Executor:  
   def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._init_executor() # 这里会执行到子类
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator: KVOutputAggregator | None = None
        
 class MultiprocExecutor:      
    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)#创建一个终结器对象，当目标对象被垃圾回收时自动调用指定函数
        self.is_failed = False
        self.failure_callback: FailureCallback | None = None

        tp_size, pp_size, pcp_size = self._get_parallel_sizes()

        # Set multiprocessing envs，内部需要看是否需要spawn
        set_multiprocessing_worker_envs()

        # use the loopback address get_loopback_ip() for communication. 最后组成tcp:port
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port()
        )
        self.rpc_broadcast_mq: MessageQueue | None = None
        scheduler_output_handle: Handle | None = None
        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        if self.parallel_config.node_rank_within_dp == 0: 
          	# node_rank_within_dp = self.node_rank % self.nnodes_within_dp 
            # 在每个数据并行组中，只有组内排名为0的节点作为主节点（leader executor）
    				# 每个数据并行组都会有自己的主多进程执行器
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            mq_connect_ip = get_ip() 
            self.rpc_broadcast_mq = MessageQueue(
                self.world_size,
                self.local_world_size,
                max_chunk_bytes=max_chunk_bytes,
                connect_ip=mq_connect_ip,
            )
            scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        # 获取多进程worker上下文
        # 常见的有: 'spawn', 'fork', 'forkserver' 三种启动方式
        context = get_mp_context()
        shared_worker_lock = context.Lock() # 多进程中见的共享锁
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            global_start_rank = (
                self.local_world_size * self.parallel_config.node_rank_within_dp
            ) # 这不是从dp = 0 即 master executor开始
						# 当父进程使用 fork()创建子进程时，子进程会继承父进程的所有文件描述符（包括socket）。
            inherited_fds: list[int] | None = (
                [] if context.get_start_method() == "fork" else None
            )

            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                #_is_driver_worker = rank % self.parallel_config.tensor_parallel_size == 0
                is_driver_worker = self._is_driver_worker(global_rank)
                unready_worker_handle = WorkerProc.make_worker_process(
                    vllm_config=self.vllm_config,
                    local_rank=local_rank,
                    rank=global_rank,
                    distributed_init_method=distributed_init_method,
                    input_shm_handle=scheduler_output_handle,
                    shared_worker_lock=shared_worker_lock,
                    is_driver_worker=is_driver_worker,
                    inherited_fds=inherited_fds,
                ) # 启动worker
                unready_workers.append(unready_worker_handle)
                if inherited_fds is not None:
                    inherited_fds.append(unready_worker_handle.death_writer.fileno())
                    inherited_fds.append(unready_worker_handle.ready_pipe.fileno())

            # Wait for all local workers to be ready.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Start background thread to monitor worker health if not in headless mode.
            if self.monitor_workers:
                self.start_worker_monitor()

            self.response_mqs = []
            # Only leader node have remote response mqs
            if self.parallel_config.node_rank_within_dp == 0:
                for rank in range(self.world_size):
                    if rank < self.local_world_size:
                        local_message_queue = self.workers[rank].worker_response_mq
                        assert local_message_queue is not None
                        self.response_mqs.append(local_message_queue)
                    else:
                        remote_message_queue = self.workers[0].peer_worker_response_mqs[
                            rank
                        ]
                        assert remote_message_queue is not None
                        self.response_mqs.append(remote_message_queue)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.

            # Wait for all input mqs to be ready.
            if self.rpc_broadcast_mq is not None:
                self.rpc_broadcast_mq.wait_until_ready()
            # Wait for all remote response mqs to be ready.
            for response_mq in self.response_mqs:
                response_mq.wait_until_ready()

            self.futures_queue = deque[tuple[FutureWrapper, Callable]]()

            self._post_init_executor()

            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                        uw.death_writer = None
                self._ensure_worker_termination([uw.proc for uw in unready_workers])

        self.output_rank = self._get_output_rank()

```





### 多进程管理

Fork方式创建进程的好处

额外设置set_multiprocessing_worker_envs，

```python
def set_multiprocessing_worker_envs():
    """Set up environment variables that should be used when there are workers
    in a multiprocessing environment. This should be called by the parent
    process before worker processes are created"""

    _maybe_force_spawn()

    # Configure thread parallelism if OMP_NUM_THREADS isn't set
    #
    # Helps to avoid CPU contention. The default of spawning a thread per
    # core combined with multiprocessing for each GPU can have a negative
    # impact on performance. The contention is amplified when running in a
    # container where CPU limits can cause throttling.
    default_omp_num_threads = 1
    if (
        "OMP_NUM_THREADS" not in os.environ
        and (current_parallelism := torch.get_num_threads()) > default_omp_num_threads
    ):
        logger.warning_once(
            "Reducing Torch parallelism from %d threads to %d to avoid "
            "unnecessary CPU contention. Set OMP_NUM_THREADS in the "
            "external environment to tune this value as needed.",
            current_parallelism,
            default_omp_num_threads,
            scope="local",
        )
        os.environ["OMP_NUM_THREADS"] = str(default_omp_num_threads)
        torch.set_num_threads(default_omp_num_threads)
```



## class WorkerProc

在创建过程中会直接调用 `make_worker_process`方法。`WorkerProc`会驱动工作进程完成4项核心任务：

1. WorkerWrapperBase.init_worker
2. WorkerWrapperBase.init_device
3. WorkerWrapperBase.load_model
4. WorkerProc.make_worker_process

```python

class WorkerProc:
   def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle: Handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool,
    ):
        self.rank = rank
        wrapper = WorkerWrapperBase(rpc_rank=local_rank, global_rank=rank)
        all_kwargs: list[dict] = [
            {} for _ in range(vllm_config.parallel_config.world_size)
        ]
        all_kwargs[local_rank] = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "is_driver_worker": is_driver_worker,
            "shared_worker_lock": shared_worker_lock,
        }
        # 工作1：初始化worker进程
        wrapper.init_worker(all_kwargs)
        self.worker = wrapper

        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )

        # 工作2：初始化设备
        self.worker.init_device() 
        # Update process title now that parallel groups are initialized
        self.setup_proc_title_and_log_prefix(
            enable_ep=vllm_config.parallel_config.enable_expert_parallel
        )
        # 工作3：加载模型
        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.worker.elastic_ep_execute("load_model")
        else:
            self.worker.load_model()

        scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = scheduler_config.async_scheduling
        if self.use_async_scheduling:
            self.async_output_queue: queue.Queue = queue.Queue()
            self.async_output_copy_thread = Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()

        # Set block size based on the attention backends
        current_platform.update_block_size_for_backend(vllm_config)

        # Initialize message queues after init_device() since multi-node setups
        # (nnodes_within_dp > 1) require distributed groups to be initialized
        # 初始化消息队列
        self._init_message_queues(input_shm_handle, vllm_config)

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()
    
    def _init_message_queues(
        self, input_shm_handle: Handle, vllm_config: VllmConfig
    ) -> None:
      	# input_shm_handle：scheduler的结果 
        if vllm_config.parallel_config.nnodes_within_dp == 1:
            # Initialize MessageQueue for receiving SchedulerOutput
            self.rpc_broadcast_mq = MessageQueue.create_from_handle(
                input_shm_handle, self.worker.rank
            )

            # Initializes a message queue for sending the model output
            self.worker_response_mq = MessageQueue(1, 1)
            self.peer_response_handles = []
        else:
            # Initialize remote MessageQueue for receiving SchedulerOutput across nodes
            self.rpc_broadcast_mq = get_inner_dp_world_group().create_mq_broadcaster(
                external_writer_handle=input_shm_handle,
                # Since there is external_writer_handle from executor proc,
                # where the ready signal from actual writer is sent out of the
                # create_mq_broadcaster method and after this setup, we make it
                # non blocking. The handshake will be triggered when
                # worker.rpc_broadcast_mq.wait_until_ready() is called
                blocking=False,
            )
            # Initializes remote message queue for sending the model output to the
            # driver worker, exposing peer_response_handles for driver worker
            # that include handles for all ranks
            self.worker_response_mq, self.peer_response_handles = (
                get_inner_dp_world_group().create_single_reader_mq_broadcasters(
                    reader_rank_in_group=0
                )
            )
        
     def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
        is_driver_worker: bool,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # Ready pipe to communicate readiness from child to parent
        ready_reader, ready_writer = context.Pipe(duplex=False)
        # Death pipe to let child detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            # Have the worker close parent end of this worker's pipes too
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )

        proc.start()
        # Close child ends of pipes here in the parent
        ready_writer.close()
        death_reader.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

```

### Init_worker

**Q1: 在 `init_worker`中创建 worker 进程时，如何保证多个进程之间互不干扰？**

这主要通过以下机制实现，而 `with set_current_vllm_config(self.vllm_config)`上下文管理器是其中的关键环节之一：

```python
with set_current_vllm_config(self.vllm_config):
    # 在工作进程初始化期间使vLLM配置可用
    self.worker = worker_class(**kwargs)
```

这个上下文管理器的作用

1. 配置作用域隔离

```python
# 在每个进程中：
# 1. 进入 with 块前：保存"旧的当前配置"
# 2. 在 with 块内：将当前进程的配置设为 self.vllm_config
# 3. 退出 with 块后：恢复为原来的配置
```

- 每个进程都有自己的 `self.vllm_config`实例
- 确保 worker 初始化时访问的是**本进程的正确配置**
- 避免多进程共用同一全局配置导致的冲突

2. **避免初始化时的配置污染**

如果没有这个机制：

- 进程A在初始化时可能意外读取到进程B的配置
- 当多个进程**并行初始化**时，配置状态会相互干扰
- 特别是当 worker 类在 `__init__`中或后续调用中依赖全局配置时



### Init_device

`init device`同`init_worker`, 需要解决多进程问题

```python
def init_device(self):
    with set_current_vllm_config(self.vllm_config):
      self.worker.init_device()  
```

`init_device`中主要完成一些环境的配置，以及多进程之间的管理，完成占卡工作，主要工作如下：

1. init_worker_distributed_environmen
2. t
       1.1 init_batch_invariance
       1.2 override_envs_for_eplb
       1.3 set_custom_all_reduce
       1.4 init_distributed_environment
       1.5 ensure_model_parallel_initialized
       1.6 ensure_ec_transfer_initialized
3. modelRunner()

```python
# gpu_worker.py
def init_device(self):
    if self.device_config.device_type == "cuda":
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and parallel_config.nnodes_within_dp == 1
        ):
            # Use local DP rank if available, otherwise use global DP rank.
            dp_local_rank = self.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = self.parallel_config.data_parallel_index

            tp_pp_world_size = (
                self.parallel_config.pipeline_parallel_size
                * self.parallel_config.tensor_parallel_size
            )

            # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
            self.local_rank += dp_local_rank * tp_pp_world_size
            assert self.local_rank < torch.accelerator.device_count(), (
                f"DP adjusted local rank {self.local_rank} is out of bounds. "
            )
            visible_device_count = (
                torch.accelerator.device_count() if torch.cuda.is_available() else 0
            )
            assert self.parallel_config.local_world_size <= visible_device_count, (
                f"local_world_size ({self.parallel_config.local_world_size}) must "
                f"be less than or equal to the number of visible devices "
                f"({visible_device_count})."
            )

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.accelerator.set_device_index(self.device)

        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Initialize the distributed environment BEFORE taking
        # memory snapshot
        # This ensures NCCL buffers are allocated before we measure
        # available memory
        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        if self.use_v2_model_runner:
            logger.info_once("Using V2 Model Runner", scope="local")

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )
    else:
        raise RuntimeError(f"Not support device type: {self.device_config.device}")

    # Initialize workspace manager
    num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
    init_workspace_manager(self.device, num_ubatches)

    # Construct the model runner
    if self.use_v2_model_runner:
        from vllm.v1.worker.gpu.model_runner import (
            GPUModelRunner as GPUModelRunnerV2,
        )

        # HACK(woosuk): This is a temporary fix to avoid type errors.
        self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
            self.vllm_config, self.device
        )
    else:
        from vllm.v1.worker.gpu_model_runner import (
            GPUModelRunner as GPUModelRunnerV1,
        )

        self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

    if self.rank == 0:
        # If usage stat is enabled, collect relevant info.
        report_usage_stats(self.vllm_config)
```



#### init_worker_distributed_environment

完成给多进程分配ip和port，保证多个进程之间可以组成processgroup，进行相互通信，1. 分配ip/port 2.初始化一个**进程组**，

#### init_distributed_environment

**ip/port管理**

```python
# 如果是多节点情况，每一个process再独立的服务器中，tcp不同
if parallel_config.nnodes > 1:
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            distributed_init_method = get_distributed_init_method(ip, port) # 拼成能用tcp url
        else: # 单节点/单机器 会出现端口竞争
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            distributed_init_method = get_distributed_init_method(ip, port)
            logger.debug(
                "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
                world_size,
                rank,
                distributed_init_method,
            )
```

```python
def get_next_dp_init_port(self) -> int:
        """
在分布式训练中，当需要初始化多个与数据并行相关的进程组时（比如worker进程和engine进程可能位于不同的进程中），为了避免端口冲突，每次初始化新的数据并行进程组时都需要获取一个新的端口。
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer
```

**初始化进程组**

```python
torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        ) 
```

backend：通信后端   "nccl"：NVIDIA GPU 通信，GPU 训练必选   "gloo"：支持 CPU 和 GPU，用于 CPU 训练或测试   "mpi"：高性能计算集群

init_method ： 初始化方法，进程如何发现彼此，现在是通过tcp地址

world_size：整个分布式训练中的**进程总数**

rank：当前进程的**唯一标识符**

#### ensure_model_parallel_initialized

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        backend: name of torch distributed communication backend.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize the model pipeline. The present function will create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks are on the same DGX box. For example if we are using 2 DGX-1 boxes with a total of 16 GPUs, rank 0 to 7 belong to the first box and ranks 8 to 15 belong to the second box.
    """	
    。。。。。
        # the layout order is: ExternalDP x DP x PP x TP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    # DP is the data parallel group that is part of the model,
    # all the ranks in the same DP group should generate simultaneously,
    # i.e. the `generate` call in the same DP group should be called together,
    # otherwise it will cause deadlock.
    # to get group_ranks for each dimension, transpose that dimension to the
    # last dimension, then reshape to 2D, then unbind the last dimension

   tp_pp_pcp_size = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * prefill_context_model_parallel_size
        )
  local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            tensor_model_parallel_size,
        )

all_ranks = torch.arange(world_size).reshape(
    -1,  # 维度0：自动计算
    data_parallel_size,  # 维度1：数据并行大小
    pipeline_model_parallel_size,  # 维度2：流水线并行大小
    prefill_context_model_parallel_size,  # 维度3：预填充上下文并行大小
    tensor_model_parallel_size,  # 维度4：张量并行大小
)

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0) # 按照零维度开始拆分
        group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # Build the DCP model-parallel groups.
    global _DCP
    assert _DCP is None, "decode context model parallel group is already initialized"
    # Note(hc): In the current implementation of decode context parallel,
    # dcp_size must not exceed tp_size, because the world size does not
    # change by DCP, it simply reuses the GPUs of TP group, and split one
    # TP group into tp_size//dcp_size DCP groups.
    group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = local_all_ranks.reshape(
            -1, decode_context_model_parallel_size
        ).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="dcp",
    )

    global _PCP
    assert _PCP is None, "prefill context parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = (
            local_all_ranks.transpose(1, 2)
            .reshape(-1, prefill_context_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
    _PCP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pcp"
    )

    # Build the pipeline model-parallel groups.
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = (
            local_all_ranks.transpose(0, 2)
            .reshape(-1, pipeline_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    global _DP
    assert _DP is None, "data parallel group is already initialized"
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        _DP = _init_stateless_group(
            group_ranks,
            "dp",
            parallel_config.data_parallel_master_ip,
            backend,
            coord_store=coord_store,
        )
    else:
        _DP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )

    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    # Don't create EP group for dense models.
    if config.model_config is None or config.model_config.is_moe:
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                data_parallel_size
                * prefill_context_model_parallel_size
                * tensor_model_parallel_size,
            )
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        if enable_elastic_ep:
            _EP = _init_stateless_group(
                group_ranks,
                "ep",
                parallel_config.data_parallel_master_ip,
                backend,
                coord_store=coord_store,
            )
        else:
            _EP = init_model_parallel_group(
                group_ranks, get_world_group().local_rank, backend, group_name="ep"
            )

        # Create EPLB group with the same ranks as EP if EPLB is enabled.
        # This is a separate process group to isolate EPLB communications
        # from MoE forward pass collectives and prevent deadlocks when
        # using torch.distributed in execution with torch.distributed in EPLB.
        global _EPLB
        assert _EPLB is None, "EPLB group is already initialized"
        if (
            config is not None
            and config.parallel_config is not None
            and config.parallel_config.enable_eplb
        ):
            if enable_elastic_ep:
                _EPLB = _init_stateless_group(
                    group_ranks,
                    "eplb",
                    parallel_config.data_parallel_master_ip,
                    backend,
                    coord_store=coord_store,
                )
            else:
                _EPLB = init_model_parallel_group(
                    group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name="eplb",
                )
    # If no EP group needed, _EP remains None
    # If no EPLB group needed, _EPLB remains None

    logger.info_once(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, PCP rank %s, "
        "TP rank %s, EP rank %s, EPLB rank %s",
        rank,
        world_size,
        _DP.rank_in_group,
        _PP.rank_in_group,
        _PCP.rank_in_group,
        _TP.rank_in_group,
        _EP.rank_in_group if _EP is not None else "N/A",
        _EPLB.rank_in_group if _EPLB is not None else "N/A",
    
```

### Load_model

```
#gpu/model_runner.py
def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
    time_before_load = time.perf_counter()
    if load_dummy_weights:
        self.load_config.load_format = "dummy"
    self.eplb.prepare_load()
    eplb_models_added = False
    with DeviceMemoryProfiler() as m:
        model_loader = get_model_loader(self.vllm_config.load_config)
        logger.info("Loading model from scratch...")

        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.vllm_config.model_config
        )
        if self.lora_config:
            self.model = self.load_lora_model(
                self.model, self.vllm_config, self.device
            )

        if self.use_aux_hidden_state_outputs:
            assert self.speculative_config is not None
            set_eagle3_aux_hidden_state_layers(self.model, self.speculative_config)
        if self.speculator is not None:
            self.speculator.load_model(self.model)
            eplb_models_added = self.eplb.maybe_register_speculator(
                self.speculator, self.speculative_config, load_dummy_weights
            )
    time_after_load = time.perf_counter()

    self.model_memory_usage = m.consumed_memory
    logger.info(
        "Model loading took %s GiB and %.6f seconds",
        format_gib(m.consumed_memory),
        time_after_load - time_before_load,
    )

    if not load_dummy_weights:
        prepare_communication_buffer_for_model(self.model)
        if self.speculator is not None:
            prepare_communication_buffer_for_model(self.speculator.model)

    # Initialize the components that require the model.
    self.model_state = init_model_state(
        self.vllm_config, self.model, self.encoder_cache, self.device
    )
    if self.is_pooling_model and self.is_last_pp_rank:
        self.pooling_runner = PoolingRunner(self.model)
    eplb_models_added |= self.eplb.maybe_register_model(
        self.model,
        self.model_config,
        load_dummy_weights,
    )
    self.eplb.maybe_start_async_loop(eplb_models_added)

    if not self.is_first_pp_rank:
        # For non-first PP ranks, create intermediate tensors sized
        # for the max capture size so they can be sliced per batch.
        # Save as persistent member so runtime can copy received data
        # into the same addresses that the CUDA graphs captured.
        self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
            batch_size=self.max_num_tokens,
            dtype=self.model_config.dtype,
            device=self.device,
        )
```

### make_worker_process

完成全部初始化工作，可以正式进入制作worker_process, 同时创建一个通信 ❓通信从workerproc到其子进程

```python
def make_worker_process(
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    input_shm_handle,  # Receive SchedulerOutput
    shared_worker_lock: LockType,
    is_driver_worker: bool,
    inherited_fds: list[int] | None = None,
) -> UnreadyWorkerProcHandle:
    context = get_mp_context()
    # Ready pipe to communicate readiness from child to parent，建立单向通道
    ready_reader, ready_writer = context.Pipe(duplex=False) 
    # Death pipe to let child detect parent process exit，建立单向通道
    death_reader, death_writer = context.Pipe(duplex=False)
    if inherited_fds is not None:
        inherited_fds = inherited_fds.copy()
        inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))#拿到一些通信底层数据，用于复用
    process_kwargs = {
        "vllm_config": vllm_config,
        "local_rank": local_rank,
        "rank": rank,
        "distributed_init_method": distributed_init_method,
        "input_shm_handle": input_shm_handle,
        "ready_pipe": ready_writer,
        "death_pipe": death_reader,
        "shared_worker_lock": shared_worker_lock,
        "is_driver_worker": is_driver_worker,
        # Have the worker close parent end of this worker's pipes too
        "inherited_fds": inherited_fds if inherited_fds is not None else [],
    }
    # Run EngineCore busy loop in background process. 启动一个子进程
    proc = context.Process(
        target=WorkerProc.worker_main,
        kwargs=process_kwargs,
        name=f"VllmWorker-{rank}",
        daemon=True,
    )

    proc.start()
    # Close child ends of pipes here in the parent
    ready_writer.close()
    death_reader.close()
    # Keep death_writer open in parent - when parent exits,
    # death_reader in child will get EOFError
    return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)
```

创建子进程`worker_main`，它与父进程之间通过`readypipe`和`deathpipe`两个管道进行通信。其中，`deathpipe`用于监控父进程是否终止，`readypipe`则用于向父进程通知子进程就绪。

```python
def worker_main(*args, **kwargs):
    """Worker initialization and execution loops.
    This runs a background process"""

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested.is_set():
            shutdown_requested.set()
            logger.debug(
                "WorkerProc handling signal %d, raising SystemExit", signum
            )
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the worker
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    worker = None
    ready_writer = kwargs.pop("ready_pipe")
    death_pipe = kwargs.pop("death_pipe", None)

    # Close inherited pipes from parent (incl. other worker pipes)
    # Explicitly passing in existing pipes and closing them makes the pipe
    # behave when using fork. Otherwise, a hidden reference to the pipes
    # exist in the child process and prevents EOF closure. 资源泄漏问题
    for fd in kwargs.pop("inherited_fds", []):
        try:
            os.close(fd)
        except Exception as e:
            logger.warning("Error closing inherited connection: %s: %s", type(e), e)

    try:
        # Initialize tracer
        rank = kwargs.get("rank", 0)
        maybe_init_worker_tracer(
            instrumenting_module_name="vllm.worker",
            process_kind="worker",
            process_name=f"Worker_{rank}",
        )

        worker = WorkerProc(*args, **kwargs)

        worker.monitor_death_pipe(death_pipe, shutdown_requested)

        # Send READY once we know everything is loaded， 发送ready信息
        ready_writer.send(
            {
                "status": WorkerProc.READY_STR,
                "handle": worker.worker_response_mq.export_handle(),
                "peer_response_handles": worker.peer_response_handles,
            }
        )

        # Ensure message queues are ready. Will deadlock if re-ordered.
        # Must be kept consistent with the Executor
        if worker.rpc_broadcast_mq is not None:
            worker.rpc_broadcast_mq.wait_until_ready()
        worker.worker_response_mq.wait_until_ready()
        ready_writer.close()
        ready_writer = None

        worker.worker_busy_loop()#启动服务一个死循环

    except Exception:
        # NOTE: if an Exception arises in busy_loop, we send
        # a FAILURE message over the MQ RPC to notify the Executor,
        # which triggers system shutdown.
        # TODO(rob): handle case where the MQ itself breaks.

        if ready_writer is not None:
            logger.exception("WorkerProc failed to start.")
        elif shutdown_requested.is_set():
            logger.info("WorkerProc shutting down.")
        else:
            logger.exception("WorkerProc failed.")

        # The parent sends a SIGTERM to all worker processes if
        # any worker dies. Set this value so we don't re-throw
        # SystemExit() to avoid zmq exceptions in __del__.
        shutdown_requested.set()

    except SystemExit as e:
        # SystemExit is raised on SIGTERM or SIGKILL, which usually indicates that
        # the graceful shutdown process did not succeed
        logger.warning("WorkerProc was terminated")
        # SystemExit must never be ignored
        raise e

    finally:
        if ready_writer is not None:
            ready_writer.close()
        if death_pipe is not None:
            death_pipe.close()
        # Clean up once worker exits busy loop
        if worker is not None:
            worker.shutdown()
            
            
            
def worker_busy_loop(self):
    """Main busy loop for Multiprocessing Workers"""
    while True:
        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(
            indefinite=True
        )
        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)

            output = func(*args, **kwargs)
        except Exception as e:
            # Notes have been introduced in python 3.11
            if hasattr(e, "add_note"):
                e.add_note(traceback.format_exc())
            logger.exception("WorkerProc hit an exception.")
            # exception might not be serializable, so we convert it to
            # string, only for logging purpose.
            if output_rank is None or self.rank == output_rank:
                self.handle_output(e)
            continue

        if output_rank is None or self.rank == output_rank:
            self.handle_output(output)
```



这个worker_busy_loop会通过rpc_messasage消息中获取到全部的方法函数，然后依次运行

方法序列是：？？？？？









# 调度问题scheduler

![img](/Users/wangjingxin/knowlege_library/大模型相关知识/VLLM/VLLM源码阅读/images/v2-eee036cd8edbc2b94d8758721b9809e8_1440w.jpg)

核心主要以下几大块：

- **在每1个推理阶段中，调度器（Scheduler）**决定哪些请求可以参与推理，并为这些请求做好逻辑块->物理块的映射。
- **在每1个推理阶段中，分布式执行者**（图中Distributed Workers部分，根据代码，我们将其命名为**model_executor**会更加合适）接收调度器传来的这些请求，分发到各个worker上去做推理。Worker中的CacheEngine负责实际管理[KV Cache](https://zhida.zhihu.com/search?content_id=242028563&content_type=Article&match_order=1&q=KV+Cache&zhida_source=entity)；Worker中的model负责加载模型、实行推理，PagedAttention相关的实现和调用就在model下。

> **这里，每1个推理阶段的定义是：prefill算1个推理阶段，每个decode各算1个推理阶段。在本文中，我们统一用step来表示“1个推理阶段”。**

入口文件依旧是LLM的generate函数

```python
class LLM:
	def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[RequestOutput]:

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )
    def _run_completion(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        output_type: type[_O],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ):
        self._add_completion_requests(
            prompts=prompts,
            params=params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            priority=priority,
            tokenization_kwargs=tokenization_kwargs,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)
     
    def _add_completion_requests(
        self,
        prompts: PromptType | Sequence[PromptType],
        params: SamplingParams
        | PoolingParams
        | Sequence[SamplingParams | PoolingParams],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> list[str]:
        seq_prompts = prompt_to_seq(prompts)#输出标准化[List]
        seq_params = self._params_to_seq(params, len(seq_prompts))#让参数和请求是一样个数的
        seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
        seq_priority = self._priority_to_seq(priority, len(prompts))

        return self._render_and_add_requests(
            prompts=(
                self._preprocess_cmpl_one(prompt, tokenization_kwargs) # prompt-> sequence group
                for prompt in maybe_tqdm(
                    seq_prompts,
                    use_tqdm=use_tqdm,
                    desc="Rendering prompts",
                )
            ),
            params=seq_params,
            lora_requests=seq_lora_requests,
            priorities=seq_priority,
        )
        
   def _run_engine(
        self,
        output_type: type[_O] | tuple[type[_O], ...],
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[_O]:
        # Run the engine.
        outputs: list[_O] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()# 进行第一步推理
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)  # type: ignore[arg-type]
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs
                            )
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s"
                            )
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
```

在送入后续流程之前，先了解一下 `self._preprocess_cmpl_one(prompt, tokenization_kwargs)`函数的功能。

## SequenceGroup

核心目的：解决一对多的输入输出映射问题，便于统一管理。

### 原生输入

在一般的推理场景中，**我们通常给模型传1个prompt及相关的采样参数**，让模型来做推理。此时你的输入可能长下面这样：

```python3
("To be or not to be,",
SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
```

**但在其余的场景中，模型decoding的策略可能更加复杂**，例如：

- **Parallel Sampling**：你传给模型1个prompt，希望模型基于这个这个prompt，给出n种不同的output
- **Beam Search**：你传给模型1个prompt，在采用Beam Search时，每个推理阶段你都会产出top k个output，其中k被称为Beam width（束宽）。

这些情况下，你传给模型的输入可能长下面这样：

```python3
# Parallel Sampling
("What is the meaning of life?",
SamplingParams(n=2, temperature=0.8, top_p=0.95, frequency_penalty=0.1))

# Beam Search (best_of = 束宽)
("It is only with the heart that one can see rightly",
SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0))
```

### SequenceGroup管理方法

![SequenceGroup](/Users/wangjingxin/knowlege_library/大模型相关知识/VLLM/VLLM源码阅读/images/image-20260328144746477-4680469.png)

- **"1个prompt -> 多个outputs"这样的结构组成一个`SequenceGroup`实例。**

- **其中每组"prompt -> output"组成一个序列（seq，属于`Sequence`实例），每个seq下有若干状态(status)属性，包括：**

  - **`WAITING`：**正在waiting队列中。waiting队列中的序列都没有做过prefill。
  - **`RUNNING`：**正在running队列中，即已经开始做推理。
  - **`SWAPPED`：**正在swapped队列中，表示此时gpu资源不足，相关的seq_group被抢占，导致其暂停推理，相关的KV block被置换到cpu上（swap out），等待gpu资源充足时再置换回来重新计算（swap in）。
  - **若干和Finish相关的状态**，表示该seq推理已经结束，具体包括：
    - **`FINISHED_STOPPED`：**正常执行完毕，例如碰到`<eos>`符号，该seq的推理正常结束了
    - **`FINISHED_LENGTH_CAPPED`**：因为seq的长度达到最大长度限制，而结束推理
    - **`FINISHED_ABORTED`**：因不正常状态，而被终止的推理。例如客户端断开连接，则服务器会终止相关seq的推理
    - **`FINISHED_IGNORED`**：因prompt过长而被终止执行的推理。本质上也是受到长度限制

  > **在vLLM中有一个重要假设：一个seq_group中的所有seq共享1个prompt。**

![SequenceGroupManager](/Users/wangjingxin/knowlege_library/大模型相关知识/VLLM/VLLM源码阅读/images/v2-274c0f1d938b868a6c2bab8a5717cb14_r.jpg)

- **在推理开始之前**，这个seq_group下只有1条seq，它就是prompt，状态为waiting。
- **在第1个推理阶段**，调度器选中了这个seq_group，由于它的采样参数中`n = 4`，所以在做完prefill之后，它会生成4个seq，它们的状态都是running。
- **在若干个推理阶段后，gpu上的资源不够了，这个seq_group不幸被调度器抢占（preemption）**，它相关的KV block也被swap out到cpu上。此时所有seq的状态变为swapped。这里要注意，**当一个seq_group被抢占时，对它的处理有两种方式：**
  - **Swap：如果该seq_group下的seq数量 > 1，此时会采取swap策略**，即把seq_group下【所有】seq的KV block从gpu上卸载到cpu上。
  - **Recomputation：如果该seq_group下的seq数量 = 1，此时会采取recomputation策略**，即把该seq_group相关的物理块都释放掉，然后将它重新放回waiting队列中。等下次它被选中推理时，就是从prefill阶段开始重新推理了，因此被称为“重计算”。（seq数量少，重新计算KV block的成本不高）

**【注意，并不是每个seq_group都会经历抢占，具体要看调度器策略和gpu资源使用情况】**

- **又过了若干个推理阶段，gpu上的资源又充足了，此时执行swap in操作**，将卸载到cpu上的KV block重新读到gpu上，继续对该seq_group做推理，此时seq的状态又变为running。
- **又过了若干个推理阶段，该seq_group中有1个seq已经推理完成了，它的状态就被标记为finish**，此后这条已经完成的seq将不参与调度。
- **又过了若干个推理阶段，这个seq_group下所有的seq都已经完成推理了**，这样就可以把它作为最终output返回了。

相信通过这个例子，我们已经能更好理解为什么vLLM要把1个prompt包装成SequenceGroup实例了。接下来我们就来看SequenceGroup实例的具体结构。



## 调度策略

现在所有的seq_group都已经被送入调度器（Scheduler）的waiting队列中了，**接下来我们就来看，在1个推理阶段中，调度器是通过什么策略来决定要送哪些seq_group去做推理的，**调度器相关的代码都在`vllm/core/scheduler.py`中

![img](/Users/wangjingxin/knowlege_library/大模型相关知识/VLLM/VLLM源码阅读/images/v2-5065184e3c2d7913856c1268b5c7da9a_1440w.jpg)

队列类型通过如下函数进行创建

```python
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
```

self.waiting，self.running，self.swapped 目标是根据调度器总策略（**FCFS**，First Come First Serve，先来先服务）原则，**对各个队列里的seq_group按照其arrival time进行排序**。



- **`BlockManager`**：**物理块管理器**。这也是vLLM自定义的一个class。截止本文写作时，vLLM提供了`BlockSpaceManagerV1`和`BlockSpaceManagerV2`两个版本的块管理器。V1是vLLM默认的版本，V2是改进版本（但还没开发完，例如不支持prefix caching等功能）。所以本文依然基于`BlockSpaceManagerV1`进行讲解。物理块管理器这个class下又维护着两个重要属性：

  - **`BlockAllocator`：物理块分配者，负责实际为seq做物理块的分配、释放、拷贝等操作。**这也是我们后文要解读的对象。其下又分成`self.gpu_allocator`和`self.cpu_allocator`两种类型，分别管理gpu和cpu上的物理块。
  - **`self.block_tables`：负责维护每个seq下的物理块列表，本质上它是一个字典，形式如`{seq_id: List[PhysicalTokenBlock]}`。**注意，这里维护者【所有】seq_group下seq的物理块，而不是单独某一个seq的。因为整个调度器都是全局的，其下的BlockManager自然也是全局的。

  

  

调度流程如下所示：





> **此时，1个推理阶段中，所有的seq_group要么全来自running队列，要么来自running + swapped队列，它们都处在decode阶段。**
>
> **至此我们要记住vLLM调度中非常重要的一点：在1个推理阶段中，所有的seq_group要么全部处在prefill阶段。要么全部处在decode阶段。**





```python
class LLMEngine：
    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    		# 目的：在正式开始推理前，执行一个虚拟批次来预热GPU和初始化CUDA上下文，避免首次推理的冷启动延迟。
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs
```





# 资源管理

目标：

1. 通信管道建立
2. 子进程失败后关闭后如何有效清理资源
3. 优雅关闭说的是什么

## 子进程关闭

```python
shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    nonlocal shutdown_requested
    if not shutdown_requested.is_set():
        shutdown_requested.set()
        logger.debug(
            "WorkerProc handling signal %d, raising SystemExit", signum
        )
        raise SystemExit()

# Either SIGTERM or SIGINT will terminate the worker
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```









# 未知

## 步骤1:入口文件server.py 

class ServeSubcommand

def cmd -》 def run_headless

```python
def cmd():
  if args.api_server_count < 1:# 情况1：无头模式（不启动API服务器）
     run_headless(args)
  elif args.api_server_count > 1:# 情况2：多个API服务器
      run_multi_api_server(args)
  else:
      # Single API server (this process).# 情况3：单个API服务器（默认）
      args.api_server_count = None
      uvloop.run(run_server(args))
```

```python
def run_headless(args: argparse.Namespace):
    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)#这里完成初始化EngineArgs所有参数初始化
    usage_context = UsageContext.OPENAI_API_SERVER 
    vllm_config = engine_args.create_engine_config(
        usage_context=usage_context, headless=True
    )# Create the VllmConfig

		...

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

		...

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler) # 优雅关闭
    signal.signal(signal.SIGINT, signal_handler) # 优雅关闭

    # nnodes_within_dp在同一个数据并行组内包含的物理节点数量。
    # node_rank_within_dp 前节点在其所属数据并行组内的局部排名。
    # node_rank_within_dp = self.node_rank % self.nnodes_within_dp （0/1）（i/n）
    if parallel_config.node_rank_within_dp > 0: # 说明并行组分布在多节点上
        from vllm.version import __version__ as VLLM_VERSION

        # Run headless workers (for multi-node PP/TP).
        host = parallel_config.master_addr
        head_node_address = f"{host}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)#启动excutor
        executor.start_worker_monitor(inline=True)
        return

    host = parallel_config.data_parallel_master_ip
    port = parallel_config.data_parallel_rpc_port
    handshake_address = get_tcp_uri(host, port)

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        local_engine_count=local_engine_count,
        start_index=vllm_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.join_first() # 等待内部进程是否有退出的
    finally:
        timeout = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)
        engine_manager.shutdown(timeout=timeout)
        logger.info("Shutting down.")
```

这里会根据你的部署环境是单节点还是多节点，采取不同的启动策略：

- **多节点场景**：通过 `MultiprocExecutor`统一拉起所有分布式执行器进程
- **单节点场景**：由 `CoreEngineProcManager`在内部管理并启动所有工作线程

## 编码操作

def _preprocess_cmpl(
    self,
    prompts: Sequence[PromptType],
    tokenization_kwargs: dict[str, Any] | None = None,
) -> Sequence[EngineInput]:
    """
    Convert prompt inputs from LLM APIs (other than [LLM.chat][]) into
    a format that can be passed to `_add_request`.

    Refer to [LLM.generate][] for a complete description of the arguments.
    
    Returns:
        A list of `EngineInput` objects ready to be passed into LLMEngine.
    """
    renderer = self.renderer
    model_config = self.model_config
    
    parsed_prompts = [
        parse_model_prompt(model_config, prompt) for prompt in prompts
    ]
    tok_params = renderer.default_cmpl_tok_params.with_kwargs(
        **(tokenization_kwargs or {})
    )
    
    return renderer.render_cmpl(parsed_prompts, tok_params)

  


def render_cmpl(
    self,
    prompts: Sequence[DictPrompt | bytes],
    tok_params: TokenizeParams | None = None,
    *,
    prompt_extras: dict[str, Any] | None = None,
):
    arrival_time = time.time()

    if tok_params is None:
        tok_params = self.default_cmpl_tok_params
    
    dict_prompts = self.render_prompts(prompts)
    tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)# 会把promts从text prompt转为 token prompt
    
    self._apply_prompt_extras(tok_prompts, prompt_extras)
    
    return [self.process_for_engine(prompt, arrival_time) for prompt in tok_prompts]

def _tokenize_singleton_prompt(
        self,
        prompt: SingletonDictPrompt,
        params: TokenizeParams,
    ) -> SingletonTokPrompt:
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt) 
            prompt = self._tokenize_prompt(prompt, params)# # 如果没有tokenid走这里，去分词

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")
    
            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]
    
        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

  

def process_for_engine(self, prompt: TokPrompt, arrival_time: float) -> EngineInput:
        engine_input: EngineInput
        if "encoder_prompt" in prompt: # 编码器-解码器模型
            engine_input = self._process_enc_dec(prompt)  # type: ignore[arg-type]
        else: # 单解码器模型
            engine_input = self._process_singleton(prompt)

        engine_input["arrival_time"] = arrival_time
    
        return engine_input

def _process_enc_dec(
        self,
        prompt: EncoderDecoderTokPrompt,
    ) -> EncoderDecoderInput:
        enc_prompt = prompt["encoder_prompt"]
        dec_prompt = prompt["decoder_prompt"]

        skip_decoder_start_token = False
        if self.mm_processor is not None:
            from vllm.multimodal.processing import EncDecMultiModalProcessor
    
            if isinstance(self.mm_processor, EncDecMultiModalProcessor):
                skip_decoder_start_token = self.mm_processor.skip_decoder_start_token
    
        return build_enc_dec_input(
            encoder_input=self._process_singleton(enc_prompt),
            decoder_input=(
                None if dec_prompt is None else self._process_singleton(dec_prompt)
            ),
            decoder_start_token_id=self.get_dec_start_token_id(),
            skip_decoder_start_token=skip_decoder_start_token,
        )
        
    def _process_singleton(self, prompt: SingletonTokPrompt) -> SingletonInput:
        if "prompt_embeds" in prompt: # 编码走这个
            return self._process_embeds(prompt)  # type: ignore[arg-type]
    			# 解码走这个
        return self._process_tokens(prompt)  # type: ignore[arg-type]