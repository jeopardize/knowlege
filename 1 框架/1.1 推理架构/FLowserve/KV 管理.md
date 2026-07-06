# KV Cache管理

Block Allocation 采用三层结构

```
BlockManager -> SubBlockManager -> BlockAloocator -> LogicalTokenBlock()
```

其中LogicalTokenBlock类似vllm，通过ref_cnt维护生命周期，只用来维护一个简单的block概念

BlockAllocator中通过维护一个链表free_blocks来进行分配和释放block

SubBlockManager：管理逻辑block -> 物理block的映射，核心数据结构

BlockManager 按照copress_ratios列表创建多个subBlockManager

p侧和d侧使用两套，原因是d侧不需要支持，prefix_cache, append_slot, fork等高级功能

swap逻辑也更简单



