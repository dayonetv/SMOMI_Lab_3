  *P??n??@l???q??@2?
{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord½2oե@!??F,?I@)½2oե@1??F,?I@:Advanced file read2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2?o?^}@!??w?oH@)??@?x@1[њ???G@:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl?`?HZ??!????}??)p??/ײ?1???T????:Preprocessing2?
eIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2O??'????!?*ֿ???)O??'????1?*ֿ???:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::PrefetchH3Mg'??!nJk??:??)H3Mg'??1nJk??:??:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2::FlatMap?n??@!N???i?I@)?}s???1mD?M????:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake?2t??!?݌1-???)l??g????1'C?T???:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::BatchV2::MemoryCache???{??!b??H|??)8j??{??1o[@?S???:Preprocessing2F
Iterator::Model????.5??!??`?.??)}?:??1/څ?#O??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??iܛ??!?50r)???)???GS=y?1(h??-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.