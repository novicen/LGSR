import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager

from time import sleep
from typing import Union, List
import logging

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_lists)):
            data, seg, data_properties = preprocessor.run_case(list_of_lists[idx],
                                                               list_of_segs_from_prev_stage_files[
                                                                   idx] if list_of_segs_from_prev_stage_files is not None else None,
                                                               plans_manager,
                                                               configuration_manager,
                                                               dataset_json)
            if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.as_tensor(np.asarray(data)).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': data_properties,
                    'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()]  # 输出到终端，也可加 FileHandler 保存到文件
# )

# def worker_wrapper(
#     list_of_files,
#     list_of_segs_from_prev_stage,
#     output_filenames_truncated,
#     plans_manager,
#     dataset_json,
#     configuration_manager,
#     queue,
#     done_event,
#     error_queue,
#     abort_event,
#     verbose
# ):
#     """子进程包装函数：捕获所有异常并记录（全局函数，可被pickle）"""
#     try:
#         # 调用原始的预处理函数
#         preprocess_fromfiles_save_to_queue(
#             list_of_files,
#             list_of_segs_from_prev_stage,
#             output_filenames_truncated,
#             plans_manager,
#             dataset_json,
#             configuration_manager,
#             queue,
#             done_event,
#             abort_event,
#             verbose
#         )
#     except Exception as e:
#         # 记录错误信息到队列
#         error_msg = f"子进程 {multiprocessing.current_process().pid} 崩溃: {str(e)}"
#         try:
#             error_queue.put(error_msg)
#         except:
#             pass  # 队列满时忽略
#         logging.error(error_msg, exc_info=True)
#         abort_event.set()  # 触发主进程终止
#     finally:
#         # 确保 done_event 置位，避免主进程死等
#         done_event.set()

# # ===================== 修复后的主函数 =====================
# def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
#                                      list_of_segs_from_prev_stage_files: Union[None, List[str]],
#                                      output_filenames_truncated: Union[None, List[str]],
#                                      plans_manager: object,
#                                      dataset_json: dict,
#                                      configuration_manager: object,
#                                      num_processes: int,
#                                      pin_memory: bool = False,
#                                      verbose: bool = False):
#     # 配置日志（输出到终端）
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(process)d - %(levelname)s - %(message)s"
#     )
    
#     # 统一使用 spawn 上下文（nnUNet 推荐）
#     context = multiprocessing.get_context('spawn')
#     # 只创建一个 Manager 实例（避免多 Manager 资源泄漏）
#     manager = Manager()
    
#     # 限制进程数不超过任务数
#     num_processes = min(len(list_of_lists), num_processes)
#     assert num_processes >= 1, "num_processes 必须 >= 1"
    
#     # 初始化进程相关变量
#     processes = []
#     done_events = []
#     target_queues = []
#     error_queues = []  # 存储子进程错误信息的队列
#     abort_event = manager.Event()  # 子进程崩溃时触发的终止事件

#     for i in range(num_processes):
#         # 为每个子进程分配任务（按进程数分片）
#         task_slice = list_of_lists[i::num_processes]
#         seg_slice = list_of_segs_from_prev_stage_files[i::num_processes] if list_of_segs_from_prev_stage_files is not None else None
#         output_slice = output_filenames_truncated[i::num_processes] if output_filenames_truncated is not None else None
        
#         # 用全局 Manager 创建队列/事件（关键：避免多 Manager）
#         done_event = manager.Event()
#         task_queue = manager.Queue(maxsize=1)
#         error_queue = manager.Queue(maxsize=1)
        
#         # 构造子进程参数（确保所有参数都能被pickle序列化）
#         worker_args = (
#             task_slice,
#             seg_slice,
#             output_slice,
#             plans_manager,
#             dataset_json,
#             configuration_manager,
#             task_queue,
#             done_event,
#             error_queue,
#             abort_event,
#             verbose
#         )
        
#         # 创建子进程（目标函数是全局的 worker_wrapper）
#         pr = context.Process(
#             target=worker_wrapper,
#             args=worker_args,
#             daemon=True
#         )
        
#         # 启动子进程并记录相关对象
#         pr.start()
#         target_queues.append(task_queue)
#         done_events.append(done_event)
#         error_queues.append(error_queue)
#         processes.append(pr)

#     # 主进程轮询获取子进程的预处理结果
#     worker_ctr = 0
#     while True:
#         # 退出条件：所有进程已完成 + 所有队列已空
#         all_done = all(event.is_set() for event in done_events)
#         all_empty = all(queue.empty() for queue in target_queues)
#         if all_done and all_empty:
#             break
        
#         # 检查是否触发终止事件（子进程报错）
#         if abort_event.is_set():
#             errors = []
#             for eq in error_queues:
#                 while not eq.empty():
#                     errors.append(eq.get())
#             raise RuntimeError(
#                 f"子进程执行失败！错误信息：{' | '.join(errors) if errors else '未知错误'}\n"
#                 "请检查：1. 数据文件路径/格式；2. 传入的对象是否可序列化；3. 内存是否充足"
#             )
        
#         # 轮询当前worker的队列
#         current_queue = target_queues[worker_ctr]
#         current_done = done_events[worker_ctr]
        
#         if not current_queue.empty():
#             # 获取预处理结果并返回
#             item = current_queue.get()
#             if pin_memory and item is not None:
#                 # 只对torch.Tensor执行pin_memory
#                 [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
#             yield item
#             # 切换到下一个worker
#             worker_ctr = (worker_ctr + 1) % num_processes
#         else:
#             # 检查当前子进程是否存活（未完成但已退出=异常）
#             current_process = processes[worker_ctr]
#             if not current_done.is_set() and not current_process.is_alive():
#                 # 收集错误信息
#                 errors = []
#                 for eq in error_queues:
#                     while not eq.empty():
#                         errors.append(eq.get())
#                 error_msg = " | ".join(errors) if errors else "子进程无响应（大概率内存不足）"
#                 raise RuntimeError(
#                     f"子进程 {current_process.pid} 意外退出！\n错误详情：{error_msg}\n"
#                     "解决方案：\n1. 将num_processes设为1；\n2. 关闭pin_memory；\n3. 检查数据是否损坏"
#                 )
#             # 短暂休眠，避免CPU空转
#             sleep(0.01)

#     # 清理资源：等待进程结束 + 关闭Manager
#     for p in processes:
#         if p.is_alive():
#             try:
#                 p.join(timeout=3)  # 超时3秒强制终止
#             except:
#                 pass
#             if p.is_alive():
#                 p.terminate()
#     manager.shutdown()

def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_segs_from_prev_stage_files[
                         i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]],
                 list_of_segs_from_prev_stage_files: Union[None, List[str]],
                 preprocessor: DefaultPreprocessor,
                 output_filenames_truncated: Union[None, List[str]],
                 plans_manager: PlansManager,
                 dataset_json: dict,
                 configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage_files is None:
            list_of_segs_from_prev_stage_files = [None] * len(list_of_lists)
        if output_filenames_truncated is None:
            output_filenames_truncated = [None] * len(list_of_lists)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files, seg_prev_stage, ofile = self._data[idx]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, data_properties = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.as_tensor(np.asarray(data))

        return {'data': data, 'data_properties': data_properties, 'ofile': ofile}


class PreprocessAdapterFromNpy(DataLoader):
    def __init__(self, list_of_images: List[np.ndarray],
                 list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                 list_of_image_properties: List[dict],
                 truncated_ofnames: Union[List[str], None],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1, verbose: bool = False):
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = \
            preprocessor, plans_manager, configuration_manager, dataset_json, truncated_ofnames

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames)),
            1, num_threads_in_multithreaded,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image, seg_prev_stage, props, ofname = self._data[idx]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg, props = self.preprocessor.run_case_npy(image, seg_prev_stage, props,
                                                   self.plans_manager,
                                                   self.configuration_manager,
                                                   self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.as_tensor(np.asarray(data))

        return {'data': data, 'data_properties': props, 'ofile': ofname}


def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for idx in range(len(list_of_images)):
            data, seg, props = preprocessor.run_case_npy(list_of_images[idx],
                                                  list_of_segs_from_prev_stage[
                                                      idx] if list_of_segs_from_prev_stage is not None else None,
                                                  list_of_image_properties[idx],
                                                  plans_manager,
                                                  configuration_manager,
                                                  dataset_json)
            list_of_image_properties[idx] = props
            if list_of_segs_from_prev_stage is not None and list_of_segs_from_prev_stage[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.as_tensor(np.asarray(data)).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   plans_manager: PlansManager,
                                   dataset_json: dict,
                                   configuration_manager: ConfigurationManager,
                                   num_processes: int,
                                   pin_memory: bool = False,
                                   verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_images), num_processes)
    assert num_processes >= 1
    target_queues = []
    processes = []
    done_events = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromnpy_save_to_queue,
                     args=(
                         list_of_images[i::num_processes],
                         list_of_segs_from_prev_stage[
                         i::num_processes] if list_of_segs_from_prev_stage is not None else None,
                         list_of_image_properties[i::num_processes],
                         truncated_ofnames[i::num_processes] if truncated_ofnames is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        done_events.append(event)
        processes.append(pr)
        target_queues.append(queue)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
