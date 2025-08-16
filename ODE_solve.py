from numpy import linspace, max
from Solvers import solver
import matplotlib.pyplot as plt

class ODE_solver(): # класс для решения систем дифференциальных уравнений

    methods = {"ERK1": 1, "ERK2": 2, "ERK4" : 4, "CROS1" : 2} # словарь, в котором указаны названия методов и порядок их точности

    def __init__(self, t_0, T, M_0=100):
        self.t_0 = t_0  # Определение t_0 для объекта
        self.T = T # Определение T для объекта
        self.M_0 = M_0 # Определение начального количества интервалов для объекта
        self.t_00 = linspace(t_0, T, M_0 + 1) # Определение начальных сеточных значений по времени для объекта
        self.y = None # Определение конечного решения системы для объекта
        self.t = None # Определение конечных сеточных значений по времени для объекта
        self.M = None # Определение конечного количества интервалов для объекта
        self.iteration = 1 # Определение количества итераций для объекта

    def __str__(self): # Магический метод для представления объекта в виде строки
        if self.M:
            result = (f"Решение получено при {self.M} точках разбиения с шагом по времени {(self.T-self.t_0)/self.M} s \n"
                      f"Максимальная погрешность согласно правилу Рунге практической оценки погрешности {max(abs(self.error))} \n"
                      f"Количество итераций до достижения сходимости = {self.iteration}")
        else:
            result = "Система дифференциальных уравнений еще не решалась"
        return result

    def ODESolve(self, f, y_t0, f_y=0, *, r=2, eps=1e-4, s=0, method="ERK4", error_graph=False, func_graph=False):

        """ f-функция производных системы дифю уравнений
        y_t0-начальное условие
        f_y - матрица Якоби необходимая для решения методом Розенброка
        r - коэффициент сгущения сетки
        eps - точность, с коротой ищется решение
        s - номер сгущения (степень, в которую возводится коэффициент сгущения)
        method - метод решения
        error_graph - аргумент, отвечающий за вывод на экран графика ошибок
        func_graph - аргумент, отвечающий за вывод на экран графиков функции с индексом func_graph или графиков всех функций """

        p = ODE_solver.methods[method] # порядок точности выбранного метода

        a = solver(f, y_t0, self.t_0, self.T, self.M_0, 2, s, method, f_y=f_y)[0] # Определение решения на начальной сетке

        self.error = 2 * eps # Задание начальной ошибки для вхождения в цикл для объекта

        i = 1 # номер сгущений

        while max(abs(self.error)) > eps: # условие на выход из цикла
            b, c, t, self.M = solver(f, y_t0, self.t_0, self.T, self.M_0, 2, s + i, method, f_y=f_y) # Определение решения на сгущенной сетке с номеро сгущения s + i
            self.error = (b - a) / (2 ** p - 1) # Определение ошибки согласно правилу Рунге практической оценки погрешности
            i += 1
            a = b
            self.iteration += 1

        if error_graph: # Вывод на экран графика ошибок
            plt.figure(f"Error")
            leg = []
            for j in range(len(y_t0)):
                plt.plot(self.t_00, self.error[:, j])
                leg.append("Error y" + r'$_{%d}$'% j)

            plt.xlabel("Контрольные точки по времени, соответствующие начальному разбиению")
            plt.ylabel('Величина ошибки согласно правилу Рунге')
            plt.legend(leg)
            plt.suptitle('Погрешность вычислений')
            plt.grid()

        self.y = c # Определение для объекта конечного решения системы после выполнения условия прекращения сгущения
        self.t = t # Определение для объекта конечных сеточных значений по времени после выполнения условия прекращения сгущения

        if func_graph == "all": # Вывод на экран графиков всех функций системы
            for j in range(len(y_t0)):
                plt.figure(f"Function y_{j}")
                plt.plot(self.t, self.y[:, j], 'k', linewidth = 2)
                plt.xlabel("t")
                plt.ylabel('y' + r'$_{%d}$'% j)
                plt.suptitle("График функции y" + r'$_{%d}$'% j)
                plt.grid()

        if func_graph is not False and func_graph != "all": # Вывод на экран графика функции с индексом func_graph
            plt.figure(f"Function y_{func_graph}")
            plt.suptitle("График функции y" + r'$_{%d}$'% func_graph)
            plt.plot(self.t, self.y[:, func_graph], 'k', linewidth = 2)
            plt.xlabel("t")
            plt.ylabel('y' + r'$_{%d}$'% func_graph)
            plt.grid()

        plt.show() # Функция, отвечающая за вывод графика на экран

        return (c, t)
