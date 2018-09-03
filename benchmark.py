
import matplotlib.pyplot as plt

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Benchmark(metaclass=Singleton):

    def update(self, N, operation_name, operation_time):
        if not hasattr(self, 'operation_to_times'):
            setattr(self, 'operation_to_times', {})

        operation_to_times = self.operation_to_times
        times = operation_to_times.get(operation_name)
        if times is None:
            operation_to_times[operation_name] = {N : (operation_time, 1)}
            return
        operation_average_time_info = times.get(N)
        if operation_average_time_info is None:
            times[N] = (operation_time, 1)
            return
        average_time = operation_average_time_info[0]
        average_count = operation_average_time_info[1]
        new_count = average_count + 1
        new_average = (float(average_time * average_count) + operation_time) / new_count
        times[N] = (new_average, new_count)

    def __repr__(self):
        return str(self.operation_to_times)

    def plot_benchmark(self):
        subplt_counter = 331
        for operation_name, times in self.operation_to_times.items():
            N_values = sorted(times.keys())
            plt.subplot(subplt_counter)
            average_times = [times[N][0] for N in N_values]
            plt.plot(N_values, average_times, 'ro', markersize=2)
            plt.title(operation_name)
            plt.grid(True)
            subplt_counter += 1

        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.45,
                        wspace=0.45)
        plt.show()


