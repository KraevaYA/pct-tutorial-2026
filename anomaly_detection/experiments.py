from re import S
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import platform
import psutil
import inspect
import warnings
import json

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)


try:
    import numba.cuda as cuda
    HAS_NUMBA_CUDA = True
except ImportError:
    HAS_NUMBA_CUDA = False


class Experiment:
    """
    A class for conducting experiments on the performance of anomaly detection algorithms.
    It allows to experiment with algorithms on CPU and GPU, compare the performance, 
    analyze algorithm parameters, and evaluate quality in the presence of labels.

    Parameters
    ----------
    cpu_alg : class, optional, default = None
        Serial algorithm class (DRAG or MERLIN).

    gpu_alg : class, optional, default = None
        Parallel algorithm class (PD3 or PALMAD).

    fixed_params : Dict, optional, default = {}
        Fixed algorithm parameters.

    varying_params : Dict, optional, default = {}
        Varying algorithm parameters.

    metrics : List[str], optional, default = ['time']
        Research metrics: 'time', 'speedup', 'phase_stats', 'accuracy'.

    results_dir : str, default = './experiments_results'
        Directory for saving experiment results.

    Attributes
    ----------
    results_: dict, default = None
        The results of experiment.

    metadata_ : dict
        A dictionary that stores the information about the experiment.
    
    cpu_info : dict
        The information about CPU.

    gpu_info : dict
        The information about GPU. 
    """

    CPU_ALGORITHMS = ['DRAG', 'MERLIN']
    GPU_ALGORITHMS = ['PD3', 'PALMAD']
    
    def __init__(self,
                cpu_alg: Optional[Any] = None,
                gpu_alg: Optional[Any] = None,
                fixed_params: Optional[Dict] = {},
                varying_params: Optional[Dict] = {},
                metrics: Optional[List[str]] = ['time'],
                results_dir: str = "./experiments_results"
                ):
        
        self.cpu_alg = cpu_alg
        self.gpu_alg = gpu_alg
        self.fixed_params = fixed_params
        self.varying_params = varying_params
        self.metrics = metrics
        self.results_dir = results_dir
        
        # Результаты
        self.results_ = None
        self.metadata_ = {
            'created_at': datetime.now().isoformat(),
            'cpu_alg': cpu_alg.__name__ if cpu_alg else None,
            'gpu_alg': gpu_alg.__name__ if gpu_alg else None,
            'fixed_params': fixed_params,
            'varying_params': varying_params,
            'metrics': metrics
        }

        self._cpu_info = self._get_cpu_info()
        self._gpu_info = self._get_gpu_info()

        # Создаем директорию для результатов
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)


    def _get_cpu_info(self) -> Dict:
        """
        Receive information about CPU.
        
        Returns
        -------
        info : Dict
            A dictionary about CPU.
        """

        info = {
            'cpu': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=True),
            'cpu_physical_cores': psutil.cpu_count(logical=False),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'platform': platform.platform(),
            'python_version': platform.python_version()
        }

        return info


    def _get_gpu_info(self) -> Dict:
        """
        Receive information about GPU.
        
        Returns
        -------
        gpu_info : Dict
            A dictionary about GPU.
        """

        gpu_info = {
            'available': False,
            'numba_installed': HAS_NUMBA_CUDA,
            'device_name': None,
            'device_count': 0,
            'compute_capability': None,
            'max_threads_per_block': None,
            'max_block_dim_x': None,
            'max_block_dim_y': None,
            'warp_size': None
        }
        
        if not HAS_NUMBA_CUDA:
            return gpu_info

        try:
            gpu_info['available'] = cuda.is_available()
            
            if gpu_info['available']:
                device = cuda.get_current_device()
                gpu_info['device_name'] = device.name
                gpu_info['device_count'] = device.id + 1
                gpu_info['compute_capability'] = device.compute_capability
                gpu_info['max_threads_per_block'] = device.MAX_THREADS_PER_BLOCK
                gpu_info['max_block_dim_x'] = device.MAX_BLOCK_DIM_X
                gpu_info['max_block_dim_y'] = device.MAX_BLOCK_DIM_Y
                gpu_info['warp_size'] = device.WARP_SIZE  
        except Exception as e:
            print(f"Ошибка при получении информации о GPU: {e}")
            
        return gpu_info


    @property
    def cpu_info(self):
        """
        Get information about CPU.
        """

        print("Информация о CPU")
        print(f"Платформа: {self._cpu_info['platform']}")
        print(f"Python: {self._cpu_info['python_version']}")
        print(f"CPU: {self._cpu_info['cpu']}")
        print(f"Ядер CPU: {self._cpu_info['cpu_cores']} (логических), "
              f"{self._cpu_info['cpu_physical_cores']} (физических)")
        print(f"ОЗУ: {self._cpu_info['ram_gb']} GB")


    @property
    def gpu_info(self):
        """
        Get information about GPU.
        """
        
        print("Информация о GPU")
        
        if not HAS_NUMBA_CUDA:
            print("Numba не установлена")
            print("Чтобы установить Numba, выполните команду pip install numba")
            return
        
        if self._gpu_info['available']:
            print(f"GPU доступен")
            print(f"Имя: {self._gpu_info['device_name']}")
            print(f"Количество устройств: {self._gpu_info['device_count']}")
            cc = self._gpu_info['compute_capability']
            print(f"Вычислительная способность: {cc[0]}.{cc[1]}")
            print(f"Макс. потоков на блок: {self._gpu_info['max_threads_per_block']}")
            print(f"Макс. размер блока: {self._gpu_info['max_block_dim_x']} x {self._gpu_info['max_block_dim_y']}")
            print(f"Размер варпа: {self._gpu_info['warp_size']}")
        else:
            print("GPU не доступен")   

    
    def can_run_gpu_experiments(self):
        """
        Check whether GPU experiments can be run.
        Displays a warning if the GPU is unavailable.

        Returns
        -------
        bool
            True if GPU experiments can be run, False otherwise.
        """
        if not HAS_NUMBA_CUDA:
            print("Numba не установлена!\n")
            print("Чтобы установить Numba, выполните команду pip install numba")
        
        if not self._gpu_info['available']:
            print("GPU не обнаружен!")
            print("Эксперименты над параллельными алгоритмами невозможно будет выполнить.\n")


    def _check_gpu_before_experiment(self) -> bool:
        """
        Check whether the experiment can be run on a GPU.

        Returns
        -------
        bool
            True if GPU experiments can be run, False otherwise.
        """

        if not self._gpu_info['available']:
            print(self._gpu_info['available'])
            print(f"GPU недоступен. Эксперименты невозможно провести.")
            return False

        return True


    def _validate_algorithm(self, expected_type: str) -> bool:
        """
        Check the algorithm class and whether it matches the expected type.
        
        Parameters
        ----------
        expected_type : str
            Expected type of the validated algorithm: 'cpu' or 'gpu'.
            
        Returns
        -------
        bool
            True if the class is valid and matches the type.
            
        Raises
        ------
        TypeError
            If the algorithm is not a class or does not match the expected type.
        """

        if (expected_type == "cpu"):
            alg_class = self.cpu_alg
        else:
            alg_class = self.gpu_alg

        if alg_class is None:
            return True
            
        # Проверка, что передан класс, а не экземпляр
        if not inspect.isclass(alg_class):
            raise TypeError(f"{expected_type.upper()} алгоритм должен быть классом, получен {type(alg_class)}")
        
        class_name = alg_class.__name__
        
        # Проверка для CPU алгоритмов
        if (expected_type == 'cpu'):
            if class_name in self.CPU_ALGORITHMS:
                return True
            elif class_name in self.GPU_ALGORITHMS:
                raise TypeError(
                    f"Класс {class_name} является параллельным алгоритмом для GPU, "
                    f"но передан как последовательный в параметре cpu_alg.\n"
                    f"Используйте параметр gpu_alg для GPU алгоритмов.\n"
                    f"Доступные CPU алгоритмы: {self.CPU_ALGORITHMS}"
                )
            else:
                # Неизвестный алгоритм
                warnings.warn(
                    f"Класс {class_name} не найден в списке известных CPU алгоритмов.\n"
                    f"Доступные CPU алгоритмы: {self.CPU_ALGORITHMS}\n",
                    UserWarning
                )
                return False
        
        # Проверка для GPU алгоритмов
        elif (expected_type == 'gpu'):
            if class_name in self.GPU_ALGORITHMS:
                return True
            elif class_name in self.CPU_ALGORITHMS:
                raise TypeError(
                    f"Класс {class_name} является последовательным алгоритмом, "
                    f"но передан как параллельный в параметре gpu_alg.\n"
                    f"Используйте параметр cpu_alg для последовательных алгоритмов.\n"
                    f"Доступные GPU-алгоритмы: {self.GPU_ALGORITHMS}"
                )
            else:
                # Неизвестный алгоритм
                warnings.warn(
                    f"Класс {class_name} не найден в списке GPU-алгоритмов.\n"
                    f"Доступные GPU-алгоритмы: {self.GPU_ALGORITHMS}\n",
                    UserWarning
                )
                return False
        
        return True


    def _validate_parameters(self, expected_type: str) -> bool:
        """
        Check parameters for algorithm.
        
        Parameters
        ----------
        expected_type : str
            Expected type of the validated algorithm: 'cpu' or 'gpu'.
            
        Returns
        -------
        bool
            True if the parameter verification is successful.
            
        Raises
        ------
        TypeError
            If required parameters are absent or if the parameters do not match the data type.
        
        ValueError
            If the parameters are outside the range of acceptable values.
        """

        if (expected_type == "cpu"):
            alg_class = self.cpu_alg
        else:
            alg_class = self.gpu_alg

        if alg_class is None:
            return True
        
        class_name = alg_class.__name__
        
        # Объединяем все параметры для проверки
        all_params = self.fixed_params.copy()
        if self.varying_params:
            varying_name = list(self.varying_params.keys())[0]
            all_params[varying_name] = None  # Временное значение для проверки
        
        # Проверка для DRAG и PD3 (требуют m и r)
        if class_name in ['DRAG', 'PD3']:
            # Проверка наличия обязательных параметров
            if 'm' not in all_params:
                raise ValueError(f"Для алгоритма {class_name} требуется параметр 'm'")
            if 'r' not in all_params:
                raise ValueError(f"Для алгоритма {class_name} требуется параметр 'r'")
            
            # Проверка значений
            if 'm' in self.fixed_params:
                m = self.fixed_params['m']
                if not isinstance(m, (int, np.integer)):
                    raise TypeError(f"Параметр 'm' должен быть целым числом, получен {type(m)}")
                if m < 3:
                    raise ValueError(f"Параметр 'm' должен быть >= 3, получено {m}")
            
            if 'r' in self.fixed_params:
                r = self.fixed_params['r']
                if not isinstance(r, (int, float, np.number)):
                    raise TypeError(f"Параметр 'r' должен быть числом, получен {type(r)}")
                if r <= 0:
                    raise ValueError(f"Параметр 'r' должен быть > 0, получено {r}")
        
        # Проверка для MERLIN и PALMAD (требуют minL, maxL, topK)
        elif class_name in ['MERLIN', 'PALMAD']:
            # Проверка наличия обязательных параметров
            if 'minL' not in all_params:
                raise ValueError(f"Для алгоритма {class_name} требуется параметр 'minL'")
            if 'maxL' not in all_params:
                raise ValueError(f"Для алгоритма {class_name} требуется параметр 'maxL'")
            if 'topK' not in all_params:
                raise ValueError(f"Для алгоритма {class_name} требуется параметр 'topK'")
            
            # Проверка значений
            if 'minL' in self.fixed_params:
                minL = self.fixed_params['minL']
                if not isinstance(minL, (int, np.integer)):
                    raise TypeError(f"Параметр 'minL' должен быть целым числом, получен {type(minL)}")
                if minL < 3:
                    raise ValueError(f"Параметр 'minL' должен быть >= 3, получено {minL}")
            
            if 'maxL' in self.fixed_params:
                maxL = self.fixed_params['maxL']
                if not isinstance(maxL, (int, np.integer)):
                    raise TypeError(f"Параметр 'maxL' должен быть целым числом, получен {type(maxL)}")
                if maxL < self.fixed_params.get('minL', 3):
                    raise ValueError(f"Параметр 'maxL' ({maxL}) должен быть >= minL ({self.fixed_params.get('minL', 3)})")
            
            if 'topK' in self.fixed_params:
                topK = self.fixed_params['topK']
                if not isinstance(topK, (int, np.integer)):
                    raise TypeError(f"Параметр 'topK' должен быть целым числом, получен {type(topK)}")
                if topK < 1:
                    raise ValueError(f"Параметр 'topK' должен быть >= 1, получено {topK}")
        
        # Проверка для n - только как изменяемый параметр
        if self.varying_params:
            varying_name = list(self.varying_params.keys())[0]
            if varying_name == 'n':
                for val in self.varying_params['n']:
                    if not isinstance(val, (int, np.integer)):
                        raise TypeError(f"Значения параметра 'n' должны быть целыми числами")
                    if val < 10:
                        raise ValueError(f"Значение 'n' = {val} должно быть >= 10")
        
        # Проверка, что n не в fixed_params
        if 'n' in self.fixed_params:
            raise ValueError("Параметр 'n' не может быть фиксированным")
        

    def _validate_params_structure(self):
        """
        Check structure of parameters for algorithm.
            
        Raises
        ------
        TypeError
            if the structure of parameters do not match the data type.

        ValueError
            If the parameters are missing.
        """

        if self.fixed_params is None and self.varying_params is None:
            raise ValueError("Должны быть указаны параметра fixed_params и varying_params")
        
        if self.fixed_params is not None:
            if not isinstance(self.fixed_params, dict):
                raise TypeError(f"fixed_params должен быть словарем, получен {type(self.fixed_params)}")
            
            # Проверка на пустые значения
            empty_params = [k for k, v in self.fixed_params.items() if v is None]
            if empty_params:
                warnings.warn(f"Параметры {empty_params} имеют значение None", UserWarning)
        
        if self.varying_params is not None:
            if not isinstance(self.varying_params, dict):
                raise TypeError(f"varying_params должен быть словарем, получен {type(self.varying_params)}")
            
            if len(self.varying_params) > 1:
                raise ValueError(f"varying_params должен содержать только один параметр, получено {len(self.varying_params)}: {list(self.varying_params.keys())}")
            
            param_name = list(self.varying_params.keys())[0]
            param_values = self.varying_params[param_name]
            
            if not isinstance(param_values, (list, np.ndarray)):
                raise TypeError(f"Значения параметра '{param_name}' должны быть списком или массивом, получен {type(param_values)}")
            
            if len(param_values) == 0:
                raise ValueError(f"Параметр '{param_name}' имеет пустой список значений")
        

    def _validate_metrics(self, has_annotation: bool) -> List[str]:
        """
        Check and correct the list of metrics.
            
        Returns
        -------
        metrics : List[str]
            Corrected list of metrics.
        """
        
        valid_metrics = ['time', 'speedup', 'phase_stats', 'accuracy']
        
        metrics = []

        # Проверка на валидность метрик
        for metric in self.metrics:
            if metric not in valid_metrics:
                warnings.warn(f"Неизвестная метрика '{metric}'. Допустимые метрики: {valid_metrics}", UserWarning)
            else:
                metrics.append(metric)
        
        # Проверка возможности вычисления speedup
        if ('speedup' in self.metrics) and (not (self.cpu_alg and self.gpu_alg)):
            warnings.warn("Метрика 'speedup' доступна только при сравнении CPU и GPU. Метрика будет пропущена.", UserWarning)
            metrics.remove('speedup')
        
        # Проверка возможности вычисления phase_stats
        if ('phase_stats' in self.metrics) and (not (self.cpu_alg or self.gpu_alg)):
            warnings.warn("Метрика 'phase_stats' требует наличия алгоритма. Метрика будет пропущена.", UserWarning)
            metrics.remove('phase_stats')
        
        # Проверка accuracy
        if ('accuracy' in self.metrics) and (not has_annotation):
            warnings.warn("Метрика 'accuracy' запрошена, но annotation не предоставлены. Метрика будет пропущена.", UserWarning)
            metrics.remove('accuracy')

        return metrics
    

    def run(self, ts: np.ndarray,
            annotation: Optional[np.ndarray] = None,
            save_results: bool = False,
            verbose: bool = False) -> Optional[pd.DataFrame]:
        """
        Run experiment.
        
        Parameters
        ----------
        ts : numpy.ndarray
            Time series.
        annotation : np.ndarray, optional, default = None
            True anomaly labeling to calculate accuracy.
        save_results : bool, default = False 
            Whether to save results.
        verbose : bool, default = False
            Output the intermediate information.
    
        Returns
        -------
        results : pd.DataFrame or None
            Experiment results.
        """

        if self.cpu_alg is None and self.gpu_alg is None:
            raise ValueError("Укажите хотя бы один алгоритм")
        
        # Проверка GPU
        if self.gpu_alg is not None and not self._check_gpu_before_experiment():
            if verbose:
                print("Эксперимент прерван: GPU недоступен")
            return None

        # Тип эксперимента
        use_cpu = self.cpu_alg is not None
        use_gpu = self.gpu_alg is not None
        is_comparison = use_cpu and use_gpu

        # Валидация классов алгоритмов и параметров
        self._validate_params_structure()
        if use_cpu:
            self._validate_algorithm("cpu")
            self._validate_parameters("cpu")
        if use_gpu:
            self._validate_algorithm("gpu")
            self._validate_parameters("gpu")

        # Проверка метрик
        has_annotation = annotation is not None
        metrics = self._validate_metrics(has_annotation)

        is_measure_accuracy = 'accuracy' in metrics
    
        if len(metrics) == 0:
            warnings.warn("Нет валидных метрик для измерения", UserWarning)
            return None
        
        if self.varying_params:
            varying_param_name = list(self.varying_params.keys())[0]
            varying_param_values = self.varying_params[varying_param_name]
            is_size_exp = (varying_param_name == 'n')

            if verbose:
                print(f"Запуск эксперимента: изменение параметра '{varying_param_name}'")
                print(f"Значения: {varying_param_values}")

        results = []
        for idx, varying_param_val in enumerate(varying_param_values):

            if verbose:
                print(f"Итерация {idx+1}: {varying_param_name} = {varying_param_val}")

            # Фиксированные параметры текущего запуска
            current_params = self.fixed_params.copy()
            if self.varying_params and varying_param_name != 'n':
                current_params[varying_param_name] = varying_param_val
            
            single_exp_results = {}
            
            # Данные
            if is_size_exp:
                ts_snippet = ts[:varying_param_val]
                true_discords = np.where(annotation[:varying_param_val] == 1)[0] if is_measure_accuracy else None
                annotation_snippet = annotation[:varying_param_val] if is_measure_accuracy else None
            else:
                ts_snippet = ts
                true_discords = np.where(annotation == 1)[0] if is_measure_accuracy else None
                annotation_snippet = annotation if is_measure_accuracy else None
            
            single_exp_results = {varying_param_name: varying_param_val}
            
            # CPU
            if use_cpu:
                cpu_alg = self.cpu_alg(**current_params)
                cpu_result = cpu_alg.predict(ts_snippet)

                single_exp_results['cpu_time'] = cpu_alg.metadata_['total_time']

                if 'phase_stats' in metrics:
                    single_exp_results['cpu_candidates'] = cpu_alg.metadata_.get('phases', {}).get('selection', {}).get('count', 0)
                    single_exp_results['cpu_discords'] = cpu_alg.metadata_.get('phases', {}).get('refinement', {}).get('count', 0)
                
                if is_measure_accuracy and true_discords is not None:
                    from anomaly_detection.metrics import get_metrics
                    from anomaly_detection.utils import select_topk_interest_discords

                    if (self.cpu_alg.__name__ == 'MERLIN'):
                        interest_cpu_result = select_topk_interest_discords(cpu_result, topK_interest=current_params['topK'])
                        #alg_metrics = get_metrics(true_discords, interest_cpu_result['indices'], interest_cpu_result['m'])
                        alg_metrics = get_metrics(annotation_snippet, interest_cpu_result['indices'], interest_cpu_result['m'])
                    else:
                        #alg_metrics = get_metrics(true_discords, cpu_result['indices'], cpu_result['m'])
                        alg_metrics = get_metrics(annotation_snippet, cpu_result['indices'], cpu_result['m'])
                        
                    
                    single_exp_results['cpu_precision'] = round(alg_metrics['Precision'], 3)
                    single_exp_results['cpu_recall'] = round(alg_metrics['Recall'], 3)
                    single_exp_results['cpu_f1'] = round(alg_metrics['F1-measure'], 3)
            
            # GPU
            if use_gpu:
                gpu_alg = self.gpu_alg(**current_params)
                gpu_result = gpu_alg.predict(ts_snippet)
                single_exp_results['gpu_time'] = gpu_alg.metadata_['total_time']
                
                if 'speedup' in metrics:
                    single_exp_results['speedup'] = single_exp_results['cpu_time'] / single_exp_results['gpu_time']
                
                if 'phase_stats' in metrics:
                    single_exp_results['gpu_candidates'] = gpu_alg.metadata_.get('phases', {}).get('selection', {}).get('count', 0)
                    single_exp_results['gpu_discords'] = gpu_alg.metadata_.get('phases', {}).get('refinement', {}).get('count', 0)
                
                if is_measure_accuracy and true_discords is not None:
                    from anomaly_detection.metrics import get_metrics
                    from anomaly_detection.utils import select_topk_interest_discords

                    if (self.gpu_alg.__name__ == 'PALMAD'):    
                        interest_gpu_result = select_topk_interest_discords(gpu_result, topK_interest=current_params['topK'])
                        #alg_metrics = get_metrics(true_discords, interest_gpu_result['indices'], interest_gpu_result['m'])
                        alg_metrics = get_metrics(annotation_snippet, interest_gpu_result['indices'], interest_gpu_result['m'])
                    else:
                        #alg_metrics = get_metrics(true_discords, gpu_result.get('indices', []), current_params['m'])
                        alg_metrics = get_metrics(annotation_snippet, gpu_result.get('indices', []), current_params['m'])
                    
                    single_exp_results['gpu_precision'] = round(alg_metrics['Precision'], 3)
                    single_exp_results['gpu_recall'] = round(alg_metrics['Recall'], 3)
                    single_exp_results['gpu_f1'] = round(alg_metrics['F1-measure'], 3)
            
            results.append(single_exp_results)

        if verbose:
            print(f"\nЭксперимент завершен.")
        
        self.results_ = pd.DataFrame(results)
        self.metadata_['metrics'] = metrics

        self.metadata_.update({
            'data_length': len(ts),
            'has_annotation': has_annotation,
            'metrics': metrics,
            'cpu_info': self._cpu_info,
            'gpu_info': self._gpu_info,
            'cpu_alg_module': self.cpu_alg.__module__ if self.cpu_alg else None,
            'gpu_alg_module': self.gpu_alg.__module__ if self.gpu_alg else None
        })

        if save_results:
            file_title = ""
            if (self.cpu_alg is not None) and (self.gpu_alg is not None): 
                file_title = f'{self.cpu_alg.__name__}_vs_{self.gpu_alg.__name__}'
            elif (self.cpu_alg is not None):
                file_title = f'{self.cpu_alg.__name__}'
            else:
                file_title = f'{self.gpu_alg.__name__}'

            now = datetime.now()
            file_title += f'_{now.strftime("%Y-%m-%d_%H:%M:%S")}'
            self.save(file_title)
        
        return self.results_
    

    def plot(self):
        """
        Plot experiment results.
        """
        
        n_plots = len(self.metadata_['metrics'])

        fig = make_subplots(
            rows=n_plots, cols=1,
            vertical_spacing=0.15
        )

        plot_row = 1
        
        if self.varying_params:
            varying_param_name = list(self.varying_params.keys())[0]
            varying_param_values = self.varying_params[varying_param_name]


        #  График времени выполнения
        if 'time' in self.metadata_['metrics']:
            
            df_time = pd.DataFrame(self.results_[varying_param_name])
            if 'cpu_time' in self.results_.columns:
                df_time['cpu_time'] = self.results_['cpu_time']
            if 'gpu_time' in self.results_.columns:
                df_time['gpu_time'] = self.results_['gpu_time']
                    
            df_time_melt = df_time.melt(id_vars=df_time.columns[0], var_name='Hardware', value_name='Value')
            column_names = list(df_time_melt.columns)

            fig1_px = px.bar(round(df_time_melt, 2), x=column_names[0], y=column_names[2], 
                                color=column_names[1], text=column_names[2], 
                                barmode='group', height=400)
            for trace in fig1_px.data:
                trace.legendgroup = "group1" 
                fig.add_trace(trace, row=plot_row, col=1)
            
            fig.update_xaxes(title_text=f'Параметр {varying_param_name}', type='category', row=plot_row, col=1)
            fig.update_yaxes(title_text='Время выполнения, сек.', row=plot_row, col=1)
            
            plot_row += 1

        
        # График ускорения
        if 'speedup' in self.metadata_['metrics']:

            fig.add_trace(
                go.Scatter(
                    x=self.results_[varying_param_name], y=self.results_['speedup'],
                    mode='lines+markers',
                    name='Ускорение',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=8, symbol='diamond'),
                    legendgroup="group2" 
                ),
                row=plot_row, col=1
            )
        
            fig.update_xaxes(title_text=f'Параметр {varying_param_name}', row=plot_row, col=1)
            fig.update_yaxes(title_text='Ускорение', row=plot_row, col=1)

            plot_row += 1
        

        # График статистики по кол-ву найденных кандидатов и диссонансов на фазах
        if 'phase_stats' in self.metadata_['metrics']:
            df_stats = pd.DataFrame(self.results_[varying_param_name])
            if 'cpu_candidates' in self.results_.columns:
                df_stats['candidates'] = self.results_['cpu_candidates']
                df_stats['discords'] = self.results_['cpu_discords']
            else:
                df_stats['candidates'] = self.results_['gpu_candidates']
                df_stats['discords'] = self.results_['gpu_discords']

            df_stats_melt = df_stats.melt(id_vars=df_stats.columns[0], var_name='Statistics', value_name='Value')
            column_names = list(df_stats_melt.columns)

            fig2_px = px.bar(round(df_stats_melt, 2), x=column_names[0], y=column_names[2], 
                                color=column_names[1], text=column_names[2], 
                                barmode='group', height=400)
            for trace in fig2_px.data:
                trace.legendgroup = "group3" 
                fig.add_trace(trace, row=plot_row, col=1)
            
            fig.update_xaxes(title_text=f'Параметр {varying_param_name}', type='category', row=plot_row, col=1)
            fig.update_yaxes(title_text='Кол-во подпоследовательностей', row=plot_row, col=1)

            plot_row += 1


        # График качества
        if 'accuracy' in self.metadata_['metrics']:
            df_metrics = pd.DataFrame(self.results_[varying_param_name])
            if 'cpu_precision' in self.results_.columns:
                df_metrics = pd.concat([df_metrics, self.results_[['cpu_precision', 'cpu_recall', 'cpu_f1']]], axis=1)
            else:
                df_metrics = pd.concat([df_metrics, self.results_[['gpu_precision', 'gpu_recall', 'gpu_f1']]], axis=1)

            df_metrics_melt = df_metrics.melt(id_vars=df_metrics.columns[0], var_name='Metrics', value_name='Value')
            column_names = list(df_metrics_melt.columns)

            fig3_px = px.bar(round(df_metrics_melt, 2), x=column_names[0], y=column_names[2], 
                                color=column_names[1], text=column_names[2], 
                                barmode='group', height=400)
            for trace in fig3_px.data:
                trace.legendgroup = "group4" 
                fig.add_trace(trace, row=plot_row, col=1)
            
            fig.update_xaxes(title_text=f'Параметр {varying_param_name}', type='category', row=plot_row, col=1)
            fig.update_yaxes(title_text='Значения метрик', row=plot_row, col=1)

        fig.update_xaxes(showgrid=False,
                        title_font=dict(size=18, color='black'),
                        linecolor='#000',
                        ticks="outside",
                        tickfont=dict(size=16, color='black'),
                        linewidth=2,
                        tickwidth=2)
        fig.update_yaxes(showgrid=False,
                        title_font=dict(size=18, color='black'),
                        linecolor='#000',
                        ticks="outside",
                        tickfont=dict(size=16, color='black'),
                        zeroline=False,
                        linewidth=2,
                        tickwidth=2)

        if (self.cpu_alg is not None) and (self.gpu_alg is not None): 
            algs_subtitle = f'{self.cpu_alg.__name__} vs {self.gpu_alg.__name__}'
            hardware_subtitle = f"CPU: {self._cpu_info['platform']},<br> GPU: {self._gpu_info['device_name']}" 
        elif (self.cpu_alg is not None):
            algs_subtitle = self.cpu_alg.__name__
            hardware_subtitle = f"CPU: {self._cpu_info['platform']}"
        else:
            algs_subtitle = self.gpu_alg.__name__
            hardware_subtitle = f"GPU: {self._gpu_info['device_name']}"

        fig.update_layout(
            title=f"Эксперимент над {algs_subtitle}: Влияние {varying_param_name}<br>{hardware_subtitle}",
            title_font=dict(size=20, color='black'),
            title_x=0.5,
            showlegend=True,
            legend=dict(
                font=dict(
                    size=16,     
                    color="black"
                )
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor='rgba(0,0,0,0)',
            legend_tracegroupgap=300,
            width=1000,
            height=300*n_plots + 100
        )
        
        fig.show(renderer="colab")
    

    def save(self, file_title: str) -> str:
        """
        Save the experiment results and metadata.
        
        Parameters
        ----------
        file_title : str
            Title of output file.
        """

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj

        if self.results_ is None:
            raise ValueError("Нет результатов для сохранения")
        
        base_path = os.path.join(self.results_dir, file_title)
        saved_files = []
        
        path = f"{base_path}.csv"
        self.results_.to_csv(path, index=False)
        saved_files.append(path)
        print(f"Результаты эксперимента сохранены в: {path}")
    
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            metadata_copy = {}
            for k, v in self.metadata_.items():
                metadata_copy[k] = convert_to_serializable(v)
            json.dump(metadata_copy, f, indent=2, ensure_ascii=False, default=str)
        saved_files.append(metadata_path)
        print(f"Метаданные сохранены в: {metadata_path}")


    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load the results and metadata of experiment from files.
        
        Parameters
        ----------
        filepath : str
            Path to csv-file with experiment results.
            
        Returns
        -------
        results : pd.DataFrame
            Uploaded experiment result.

        Raises
        ------
        ImportError
            If Failed to import algorithm class.
        """

        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            self.results_ = pd.read_csv(filepath)
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
        
        print(f"Загружено {len(self.results_)} записей из {filepath}")

        metadata_path = filepath.replace(ext, '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                loaded_metadata = json.load(f)
            print(f"Загружены метаданные из {metadata_path}")
        
        self.fixed_params = loaded_metadata.get('fixed_params', self.fixed_params)
        self.varying_params = loaded_metadata.get('varying_params', self.varying_params)
        self.metrics = loaded_metadata.get('metrics', self.metrics)
        
        cpu_alg_name = loaded_metadata.get('cpu_alg')
        cpu_alg_module = loaded_metadata.get('cpu_alg_module')

        if cpu_alg_name:
            try:
                module = __import__(cpu_alg_module, fromlist=[cpu_alg_name])
                self.cpu_alg = getattr(module, cpu_alg_name)
            except ImportError as e:
                print(f"Не удалось импортировать {cpu_alg_name}: {e}")

        gpu_alg_name = loaded_metadata.get('gpu_alg')
        gpu_alg_module = loaded_metadata.get('gpu_alg_module')

        if (gpu_alg_name):
            try:
                module = __import__(gpu_alg_module, fromlist=[gpu_alg_name])
                self.gpu_alg = getattr(module, gpu_alg_name)
            except ImportError as e:
                print(f"Не удалось импортировать {gpu_alg_name}: {e}")
        
        self.metadata_ = loaded_metadata

        return self.results_