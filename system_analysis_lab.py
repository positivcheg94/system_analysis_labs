#!/usr/bin/python

import tkinter as tk
import tkinter.filedialog as file_dialog
import tkinter.messagebox as message_box
from collections import defaultdict
from os import path

import pandas
import numpy as np
from constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from openpyxl import Workbook

from functional_restoration.model.additive_model import Additive, AdditiveDegreeFinder
from functional_restoration.model.multiplicative_model import Multiplicative, MultiplicativeDegreeFinder
from functional_restoration.model.mixed_model import Mixed
from functional_restoration.private.shared import transform_independent_x_matrix
from risk_prediction import bulk_predict

CONST_LIMIT = 1000


def __validate_only_digits__(S):
    return S.isdigit()


def __pick_save_file_dialog__(callback=None):
    picked_file = file_dialog.asksaveasfile()
    if picked_file is None:
        return None
    picked_file_name = picked_file.name
    try:
        callback(picked_file_name)
    except:
        print('callack exception occured')
    return picked_file_name


def __parse_file__(file):
    data = pandas.ExcelFile(file, dtype=DEFAULT_FLOAT_TYPE).parse()
    columns = data.keys().tolist()
    tmp_dict = defaultdict(list)
    for column in sorted(columns):
        tmp_dict[column[:2].lower()].append(data[column].tolist())

    x = []
    y = []
    for i in sorted(tmp_dict.keys()):
        if i[0] == 'x':
            x.append(tmp_dict[i])
        elif i[0] == 'y':
            y.append(tmp_dict[i][0])

    if len(x) > 3:
        print("sry, it's only experimental program, you can not have more than 3 x-variables")
        raise Exception

    x_matrix = np.array(x)
    y_matrix = np.array(y)

    return x_matrix, y_matrix


def __risks_parse_file__(file):
    data = pandas.ExcelFile(file, dtype=DEFAULT_FLOAT_TYPE).parse()
    columns = data.keys().tolist()
    data_dict = {}
    for column in sorted(columns):
        data_dict[column.lower()] = data[column].tolist()
    return data_dict


class Application:
    @staticmethod
    def __show_error__(title, message):
        message_box.showerror(title, message)

    @staticmethod
    def __close_windows__(*windows):
        for window in windows:
            window.destroy()

    def __init__(self, main_window=tk.Tk(), risks_window=tk.Tk()):
        risks_window.withdraw()

        self._main_window = main_window
        self._main_window.title('System analysis')
        self._risks_window = risks_window
        self._risks_window.title('Risks')

        self._main_window.protocol("WM_DELETE_WINDOW",
                                   lambda: Application.__close_windows__(self._main_window, self._risks_window))
        self._risks_window.protocol("WM_DELETE_WINDOW",
                                    lambda: Application.__close_windows__(self._main_window, self._risks_window))

        self._validator = self._main_window.register(__validate_only_digits__)
        self._risks_validator = self._risks_window.register(__validate_only_digits__)
        self.__init_widgets__()
        # self.set_size(600,480)
        # self.resizeable(False)

    def __init_widgets__(self):

        self.y1_abnormal = 11.7
        self.y1_crash = 10.5
        self.y2_abnormal = 1
        self.y2_crash = 0
        self.y3_abnormal = 11.85
        self.y3_crash = 10.5

        self._last_plots = None

        self._data = None
        self._risks_data = None

        self._last_result = None

        self._method = tk.StringVar()
        self._method.set(DEFAULT_METHOD)

        self._form = tk.StringVar()
        self._form.set(DEFAULT_FORM)

        self._polynom_var = tk.IntVar()
        self._polynom_var.set(1)

        self._weights = tk.IntVar()
        self._weights.set(1)

        self._find_lambdas = tk.IntVar()
        self._find_lambdas.set(1)

        self._find_best_degree = tk.IntVar()
        self._find_best_degree.set(0)

        self._top_block = tk.Frame(self._main_window)
        self._middle_block = tk.Frame(self._main_window)
        self._bottom_block = tk.Frame(self._main_window)

        self._input_file_name = None
        self._result_file_name = None
        self._risks_input_file_name = None
        self._risks_result_file_name = None

        self._risks_current_step = 0

        self._plot1 = None
        self._plot2 = None
        self._plot3 = None

        #        # input_data_frame
        self._input_data_frame = tk.Frame(self._top_block, padx=5, relief='groove', borderwidth=5)
        #            # sample select frame
        self._sample_select_frame = tk.Frame(self._input_data_frame, pady=20)
        self._sample_select_frame_label = tk.Label(self._input_data_frame, text='Input data')
        #                # sample size frame

        self._sample_size_frame = tk.Frame(self._sample_select_frame)
        self._sample_size_frame_label = tk.Label(self._sample_size_frame, text='Sample size')
        self._sample_size_frame_entry = tk.Entry(self._sample_size_frame, validate='key',
                                                 validatecommand=(self._validator, '%S'))
        self._sample_size_frame_entry.insert(0, '50')
        self._sample_size_frame_label.pack(side='left')
        self._sample_size_frame_entry.pack(side='right')
        self._sample_size_frame.pack()
        #                # !sample size frame
        #                # files picker frame
        self._files_picker_frame = tk.Frame(self._sample_select_frame)
        #                    # input file picker frame
        self._files_picker_input_frame = tk.Frame(self._files_picker_frame)
        self._files_picker_input_label = tk.Label(self._files_picker_input_frame, text='select file')
        self._files_picker_input_button = tk.Button(self._files_picker_input_frame, text='...',
                                                    command=self.__select_input_file__)
        self._files_picker_input_label.pack(side='left')
        self._files_picker_input_button.pack(side='right')
        #                    # !input file picker frame
        #                    # result file picker frame
        self._files_picker_result_frame = tk.Frame(self._files_picker_frame)
        self._files_picker_result_label = tk.Label(self._files_picker_result_frame, text='select file')
        self._files_picker_result_button = tk.Button(self._files_picker_result_frame, text='...',
                                                     command=lambda: __pick_save_file_dialog__(
                                                             self.__set_result_file_name__))
        self._files_picker_result_label.pack(side='left')
        self._files_picker_result_button.pack(side='right')
        #                    # !result file picker frame
        self._files_picker_input_frame.pack(fill='x')
        self._files_picker_result_frame.pack(fill='x')
        #                # !files picker frame
        self._sample_select_frame_label.pack()
        self._sample_select_frame.pack()
        self._files_picker_frame.pack()
        #            # !sample select frame
        #            # vectors frame
        self._vectors_frame = tk.Frame(self._input_data_frame, pady=20)
        #                #first vector frame
        self._vector_x1_frame = tk.Frame(self._vectors_frame, pady=5)
        self._vector_x1_name = tk.Label(self._vector_x1_frame, text='Dimension of X1').pack(side='left')
        self._vector_x1_dimension = tk.Label(self._vector_x1_frame, bg='white', width=5)
        self._vector_x1_dimension.pack(side='right')
        #                #!first vector frame
        #                #second vector frame
        self._vector_x2_frame = tk.Frame(self._vectors_frame, pady=5)
        self._vector_x2_name = tk.Label(self._vector_x2_frame, text='Dimension of X2').pack(side='left')
        self._vector_x2_dimension = tk.Label(self._vector_x2_frame, bg='white', width=5)
        self._vector_x2_dimension.pack(side='right')
        #                #!second vector frame
        #                #third vector frame
        self._vector_x3_frame = tk.Frame(self._vectors_frame, pady=5)
        self._vector_x3_name = tk.Label(self._vector_x3_frame, text='Dimension of X3').pack(side='left')
        self._vector_x3_dimension = tk.Label(self._vector_x3_frame, bg='white', width=5)
        self._vector_x3_dimension.pack(side='right')
        #                #!third vector frame
        #                #result vector frame
        self._vector_y_frame = tk.Frame(self._vectors_frame, pady=5)
        self._vector_y_name = tk.Label(self._vector_y_frame, text='Dimension of Y').pack(side='left')
        self._vector_y_dimension = tk.Label(self._vector_y_frame, bg='white', width=5)
        self._vector_y_dimension.pack(side='right')
        #                #!result vector frame
        self._vector_x1_frame.pack(fill='x')
        self._vector_x2_frame.pack(fill='x')
        self._vector_x3_frame.pack(fill='x')
        self._vector_y_frame.pack(fill='x')
        #            #!vectors frame
        self._sample_select_frame.pack(fill='x')
        self._vectors_frame.pack(fill='x')
        #        #!input_data_frame
        self._input_data_frame.pack(side='left')

        #        #polynoms frame
        self._polynoms_frame = tk.Frame(self._top_block, padx=5, relief='groove', borderwidth=5)
        #            #polynoms appearance
        self._polynoms_appearence_frame = tk.Frame(self._polynoms_frame, relief='groove', borderwidth=2)
        tk.Label(self._polynoms_appearence_frame, text='Polynom appearence').pack(fill='x')
        self._polymon_chebyshev = tk.Radiobutton(self._polynoms_appearence_frame, text="Chebyshev's polynoms",
                                                 variable=self._polynom_var, value=1)
        self._polymon_legendre = tk.Radiobutton(self._polynoms_appearence_frame, text="Legendre's polynoms",
                                                variable=self._polynom_var, value=2)
        self._polymon_lagger = tk.Radiobutton(self._polynoms_appearence_frame, text="Lagger's polynoms",
                                              variable=self._polynom_var, value=3)
        self._polymon_hermite = tk.Radiobutton(self._polynoms_appearence_frame, text="Hermite's polynoms",
                                               variable=self._polynom_var, value=4)
        self._polymon_chebyshev.pack(anchor='w')
        self._polymon_legendre.pack(anchor='w')
        self._polymon_lagger.pack(anchor='w')
        self._polymon_hermite.pack(anchor='w')
        self._polynoms_appearence_frame.pack()
        #            #!polynoms appearance
        #            #degree of polymonial

        self._degree_of_polynomial_frame = tk.Frame(self._polynoms_frame, relief='groove', borderwidth=2)
        #               #find best degree frame
        self._find_best_degree_frame = tk.Frame(self._degree_of_polynomial_frame)
        tk.Checkbutton(self._find_best_degree_frame, variable=self._find_best_degree,
                       text='Find best degree up to selected').pack(fill='both')
        self._find_best_degree_frame.pack(fill='x')
        #               #!find best degree frame
        tk.Label(self._degree_of_polynomial_frame, text='Degree of polynomial').pack(fill='x')
        #                #spinbox 1 frame
        self._spinbox1_frame = tk.Frame(self._degree_of_polynomial_frame)
        tk.Label(self._spinbox1_frame, text='Degree of X1').pack(side='left')
        v = tk.IntVar()
        v.set(7)
        self._degree_of_x1 = tk.Spinbox(self._spinbox1_frame, from_=1, to=CONST_LIMIT, textvariable=v, width=3,
                                        state='readonly')
        self._degree_of_x1.pack(side='right')
        self._spinbox1_frame.pack(fill='x')
        #                #!spinbox 1 frame
        #                #spinbox 2 frame
        self._spinbox2_frame = tk.Frame(self._degree_of_polynomial_frame)
        tk.Label(self._spinbox2_frame, text='Degree of X2').pack(side='left')
        v = tk.IntVar()
        v.set(7)
        self._degree_of_x2 = tk.Spinbox(self._spinbox2_frame, from_=1, to=CONST_LIMIT, textvariable=v, width=3,
                                        state='readonly')
        self._degree_of_x2.pack(side='right')
        self._spinbox2_frame.pack(fill='x')
        #                #!spinbox 2 frame
        #                #spinbox 3 frame
        self._spinbox3_frame = tk.Frame(self._degree_of_polynomial_frame)
        tk.Label(self._spinbox3_frame, text='Degree of X3').pack(side='left')
        v = tk.IntVar()
        v.set(10)
        self._degree_of_x3 = tk.Spinbox(self._spinbox3_frame, from_=1, to=CONST_LIMIT, textvariable=v, width=3,
                                        state='readonly')
        self._degree_of_x3.pack(side='right')
        self._spinbox3_frame.pack(fill='x')
        #                #!spinbox 3 frame
        self._degree_of_polynomial_frame.pack(fill='x')
        #            #degree of polymonial
        #        #!polynoms frame
        self._polynoms_frame.pack(side='left')

        #        #global additional frame
        self._global_additional_frame = tk.Frame(self._top_block, padx=5, relief='groove', borderwidth=5)
        #            #additional frame
        self._additional_frame = tk.Frame(self._global_additional_frame, relief='groove', borderwidth=2)
        tk.Label(self._additional_frame, text='Additional options').pack(fill='x')
        #                #weights frame
        self._weights_frame = tk.Frame(self._additional_frame, relief='groove', borderwidth=2)
        tk.Radiobutton(self._weights_frame, text="Average", variable=self._weights, value=1).pack(fill='x')
        tk.Radiobutton(self._weights_frame, text="Min and Max", variable=self._weights, value=2).pack(fill='x')
        self._weights_frame.pack(fill='x')
        #                #!weights frame
        self._find_lambdas_checkbutton = tk.Checkbutton(self._additional_frame, text='Find lambda matrix in 3d systems',
                                                        variable=self._find_lambdas)
        self._find_lambdas_checkbutton.pack(fill='x')
        #                # pick epsilon
        self._pick_epsilon_frame = tk.Frame(self._additional_frame)
        tk.Label(self._pick_epsilon_frame, text='eps').pack(side='left')
        self._pick_epsilon_edit = tk.Entry(self._pick_epsilon_frame, width=5)
        self._pick_epsilon_edit.insert(0, '1e-6')
        self._pick_epsilon_edit.pack(side='right')
        self._pick_epsilon_frame.pack(anchor='n')
        #                #! pick epsilon
        #                # pick method
        self._pick_method_frame = tk.Frame(self._additional_frame, relief='groove', borderwidth=2)
        tk.Label(self._pick_method_frame, text='Optimization method').pack(side='left')
        self._pick_method = tk.OptionMenu(self._pick_method_frame, self._method, *tuple(OPTIMIZATION_METHODS))
        self._pick_method.pack(side='right')
        self._pick_method_frame.pack(anchor='n')
        #                # pick method
        #                # pick form
        self._pick_form_frame = tk.Frame(self._additional_frame, relief='groove', borderwidth=2)
        tk.Label(self._pick_form_frame, text='Form').pack(side='left')
        self._pick_form = tk.OptionMenu(self._pick_form_frame, self._form, *tuple(FORMS))
        self._pick_form.pack(side='right')
        self._pick_form_frame.pack(anchor='n')
        #                # pick form
        self._additional_frame.pack(fill='x')
        #        #!global additional frame
        self._global_additional_frame.pack(side='left', fill='x')

        #        #
        #        # Top block pack
        #        #
        self._top_block.pack(fill='x')

        #        #left button
        self._middle_frame_button_left = tk.Frame(self._middle_block)
        self._plot_button = tk.Button(self._middle_frame_button_left, command=self._make_plot, text='Make plots')
        self._plot_button.pack()
        self._change_mode_to_risks = tk.Button(self._middle_block, command=self._switch_to_risks, text='Risks')
        self._change_mode_to_risks.pack()
        #        #right button
        self._middle_frame_button_right = tk.Frame(self._middle_block)
        self._process_button = tk.Button(self._middle_frame_button_right, text='Process calculations',
                                         command=self.__make_calculations__)
        self._process_button.pack()

        self._middle_frame_button_left.pack(side='left', expand=True)
        self._middle_frame_button_right.pack(side='right', expand=True)
        self._middle_block.pack(fill='x')

        self._result_frame = tk.Frame(self._bottom_block)
        self._result_window_scrollbar = tk.Scrollbar(self._result_frame)
        self._result_window_scrollbar.pack(side='right', fill='y')
        self._result_window = tk.Text(self._result_frame, state='disabled',
                                      yscrollcommand=self._result_window_scrollbar.set)
        self._result_window.pack(fill='both')
        self._result_window_scrollbar.config(command=self._result_window.yview)
        self._result_frame.pack(fill='x')
        self._bottom_block.pack(fill='x')

        # risks #
        # risks #
        # risks #

        self._risks_picker_frame = tk.Frame(self._risks_window)

        self._risks_files_picker_input_frame = tk.Frame(self._risks_picker_frame)
        self._risks_files_picker_input_label = tk.Label(self._risks_files_picker_input_frame, text='select file')
        self._risks_files_picker_input_button = tk.Button(self._risks_files_picker_input_frame, text='...',
                                                          command=self.__risks_select_input_file__)
        self._risks_files_picker_input_label.pack(side='left')
        self._risks_files_picker_input_button.pack(side='right')

        self._risks_files_picker_result_frame = tk.Frame(self._risks_picker_frame)
        self._risks_files_picker_result_label = tk.Label(self._risks_files_picker_result_frame, text='select file')
        self._risks_files_picker_result_button = tk.Button(self._risks_files_picker_result_frame, text='...',
                                                           command=lambda: __pick_save_file_dialog__(
                                                                   self.__risks_set_result_file_name__))
        self._risks_files_picker_result_label.pack(side='left')
        self._risks_files_picker_result_button.pack(side='right')

        self._risks_files_picker_input_frame.pack(fill='x')
        self._risks_files_picker_result_frame.pack(fill='x')

        self._step_size_frame = tk.Frame(self._risks_picker_frame)
        tk.Label(self._step_size_frame, text='step size').pack(side='left')
        self._step_size_edit = tk.Entry(self._step_size_frame, validate='key',
                                        validatecommand=(self._risks_validator, '%S'))
        self._step_size_edit.insert(0, '10')
        self._step_size_edit.pack(side='right')
        self._step_size_frame.pack(fill='x')

        self._risks_picker_frame.pack(side='left')

        self._risks_plots_frame = tk.Frame(self._risks_window)
        figure = Figure()
        self._plot1 = figure.add_subplot(3, 1, 1)
        self._plot2 = figure.add_subplot(3, 1, 2)
        self._plot3 = figure.add_subplot(3, 1, 3)
        self._risks_canvas = FigureCanvasTkAgg(figure, master=self._risks_plots_frame)
        self._risks_canvas.show()
        self._risks_canvas.get_tk_widget().pack(fill='both')
        self._risks_plots_frame.pack()

        self._return_to_main_button_frame = tk.Frame(self._risks_window)
        tk.Button(self._return_to_main_button_frame, command=self._switch_to_main_window,
                  text='Return to main window').pack()
        self._return_to_main_button_frame.pack(fill='x')

        self._process_risk_calculations_frame = tk.Frame(self._risks_window)
        tk.Button(self._process_risk_calculations_frame, command=self.__risks_make_calculations__,
                  text='Process risk calcucations').pack()
        self._process_risk_calculations_frame.pack(fill='x')

    def __make_calculations__(self):
        if self._input_file_name is None or self._result_file_name is None:
            self.__show_error__('Open File Error', 'You did not pick the files')
            return
        find_best_degrees = bool(self._find_best_degree.get())
        # samples = int(self._sample_size_frame_entry.get())
        polynom = 'chebyshev'
        p = self._polynom_var.get()
        if p == 1:
            polynom = 'chebyshev'
        elif p == 2:
            polynom = 'legendre'
        elif p == 3:
            polynom = 'laguerre'
        else:
            polynom = 'hermite'
        degree_x1 = int(self._degree_of_x1.get())
        degree_x2 = int(self._degree_of_x2.get())
        degree_x3 = int(self._degree_of_x3.get())
        degrees = [degree_x1, degree_x2, degree_x3]

        eps = None
        try:
            eps = float(self._pick_epsilon_edit.get())
        finally:
            pass

        if self._weights.get() == 1:
            weights = 'average'
        else:
            weights = 'minmax'
        find_lambda = bool(self._find_lambdas.get())

        method = OPTIMIZATION_METHODS[self._method.get()]
        form = self._form.get()

        try:

            if form == 'mul':
                if find_best_degrees:
                    res = MultiplicativeDegreeFinder(degrees, weights, method, polynom, find_lambda).fit(*self._data)
                    results = res.text()
                    self._last_plots = res.plot
                else:
                    res = Multiplicative(degrees, weights, method, polynom, find_lambda).fit(*self._data)
                    results = res.text()
                    self._last_plots = res.plot
            elif form == 'mul-add':
                res = Mixed(degrees, weights, method, ['mul', 'add'], polynom, find_lambda).fit(*self._data)
                results = res.text()
                self._last_plots = res.plot
            else:
                if find_best_degrees:
                    res = AdditiveDegreeFinder(degrees, weights, method, polynom, find_lambda).fit(*self._data)
                    results = res.text()
                    self._last_plots = res.plot
                else:
                    res = Additive(degrees, weights, method, polynom, find_lambda).fit(*self._data)
                    results = res.text()
                    self._last_plots = res.plot

            self.reset_and_insert_results(results)
            self.__write_to_file__(results)
        except ValueError as v_error:
            self.__show_error__('ValueError', str(v_error))

    def __risks_make_calculations__(self):

        risk_div1 = self.y1_crash - self.y1_abnormal
        risk_div2 = self.y2_crash - self.y2_abnormal
        risk_div3 = self.y3_crash - self.y3_abnormal

        n = len(self._risks_data['q'])
        self._risks_data['x3'] = (self._risks_data['x3'] + np.random.randn(n) * 1e-8).tolist()
        lag_len = 70
        prediction_length = int(self._step_size_edit.get())
        times = round(n / prediction_length)

        output_data = []

        y1 = []
        y2 = []
        y3 = []

        for i in range(times-10):
            start = prediction_length * i
            end = start + lag_len
            super_end = end + prediction_length

            next_time = self._risks_data['q'][end:super_end]

            current_x1 = [self._risks_data['x1'][start:end], self._risks_data['x2'][start:end],
                          self._risks_data['x3'][start:end], self._risks_data['x4'][start:end]]

            current_x2 = [self._risks_data['x2'][start:end], self._risks_data['x3'][start:end]]

            current_x3 = [self._risks_data['x2'][start:end], self._risks_data['x3'][start:end],
                          self._risks_data['x4'][start:end]]

            current_y1 = self._risks_data['y1'][start:end]
            current_y2 = self._risks_data['y2'][start:end]
            current_y3 = self._risks_data['y3'][start:end]

            processing_model_x1 = Multiplicative([20, 20, 20, 20], 'average', 'lstsq', find_split_lambdas=True)
            processing_model_x2 = Multiplicative([20, 20, ], 'average', 'lstsq', find_split_lambdas=True)
            processing_model_x3 = Multiplicative([20, 20, 20], 'average', 'lstsq', find_split_lambdas=True)

            # next_x1 = bulk_predict(current_x1, prediction_length)
            # next_x2 = bulk_predict(current_x2, prediction_length)
            # next_x3 = bulk_predict(current_x3, prediction_length)

            next_x1 = [self._risks_data['x1'][end:super_end], self._risks_data['x2'][end:super_end],
                       self._risks_data['x3'][end:super_end], self._risks_data['x4'][end:super_end]]

            next_x2 = [self._risks_data['x2'][end:super_end], self._risks_data['x3'][end:super_end]]

            next_x3 = [self._risks_data['x2'][end:super_end], self._risks_data['x3'][end:super_end],
                       self._risks_data['x4'][end:super_end]]

            res_x1 = processing_model_x1.fit(transform_independent_x_matrix(current_x1), [current_y1])
            res_x2 = processing_model_x2.fit(transform_independent_x_matrix(current_x2), [current_y2])
            res_x3 = processing_model_x3.fit(transform_independent_x_matrix(current_x3), [current_y3])

            # """
            next_y1 = res_x1.predict(transform_independent_x_matrix(next_x1), normalize=True).flatten().tolist()
            next_y2 = res_x2.predict(transform_independent_x_matrix(next_x2), normalize=True).flatten().tolist()
            next_y3 = res_x3.predict(transform_independent_x_matrix(next_x3), normalize=True).flatten().tolist()

            y1 = y1 + next_y1
            y2 = y2 + next_y2
            y3 = y3 + next_y3
            # """

            """
            next_y1 = self._risks_data['y1'][end:super_end]
            next_y2 = self._risks_data['y2'][end:super_end]
            next_y3 = self._risks_data['y3'][end:super_end]

            y1 = y1+next_y1
            y2 = y2+next_y2
            y3 = y3+next_y3
            """

            self.draw_plot1(y1)
            self.draw_plot2(y2)
            self.draw_plot3(y3)

            for i in range(prediction_length):
                tmp_list = [int(next_time[i]), next_y1[i], next_y2[i], next_y3[i]]

                if next_y1[i] > self.y1_abnormal and next_y2[i] > self.y2_abnormal and next_y3[i] > self.y3_abnormal:
                    tmp_list.append('система функционирует нормально')
                    state = 'ok'
                    total_risk = 0
                    danger_level = 0
                elif next_y1[i] > self.y1_crash and next_y2[i] > self.y2_crash and next_y3[i] > self.y2_crash:
                    f1 = next_y1[i] <= self.y1_abnormal
                    f2 = next_y2[i] <= self.y2_abnormal
                    f3 = next_y3[i] <= self.y3_abnormal
                    abnormal_sum = sum((f1, f2, f3))
                    tmp_list.append('система функционирует нештатно')
                    state = 'bad'
                    risk_1 = (next_y1[i] - self.y1_abnormal) / risk_div1
                    risk_2 = (next_y2[i] - self.y2_abnormal) / risk_div2
                    risk_3 = (next_y3[i] - self.y3_abnormal) / risk_div3
                    total_risk = 1 - (1 - risk_1) * (1 - risk_2) * (1 - risk_3)
                    if total_risk>0.2:
                        danger_level = 2+total_risk*5
                    else:
                        if abnormal_sum is 1:
                            danger_level = 1
                        else:
                            danger_level = 2

                else:
                    tmp_list.append('система функционирует аварийно')
                    state = 'so bad'
                    total_risk = 1
                    danger_level = 7

                tmp_list.append(total_risk)
                if state != 'ok':
                    tmp_list.append('insert reason here LOL')
                else:
                    tmp_list.append(' - ')
                tmp_list.append(danger_level)

                output_data.append(tmp_list)

            wb = Workbook()
            ws = wb.active
            for i in output_data:
                ws.append(i)
            wb.save('output.xlsx')

    def draw_plot1(self, x):
        n = len(x)
        self._plot1.clear()
        self._plot1.set_ylim(10, 15)
        self._plot1.axhline(self.y1_abnormal, color='b')
        self._plot1.axhline(self.y1_crash, color='r')
        self._plot1.plot(range(n), x, 'y')
        self._risks_canvas.draw()

    def draw_plot2(self, x):
        n = len(x)
        self._plot2.clear()
        self._plot2.axhline(self.y2_abnormal, color='b')
        self._plot2.axhline(self.y2_crash, color='r')
        self._plot2.plot(range(n), x, 'y')
        self._risks_canvas.draw()

    def draw_plot3(self, x):
        n = len(x)
        self._plot3.clear()
        self._plot3.set_ylim(10, 15)
        self._plot3.axhline(self.y3_abnormal, color='b')
        self._plot3.axhline(self.y3_crash, color='r')
        self._plot3.plot(range(n), x, 'y')
        self._risks_canvas.draw()

    def __set_input_file_name__(self, file_name):
        self._input_file_name = file_name
        self._files_picker_input_label.config(text=path.basename(file_name))

    def __risks_set_input_file_name__(self, file_name):
        self._risks_input_file_name = file_name
        self._risks_files_picker_input_label.config(text=path.basename(file_name))

    def __set_result_file_name__(self, file_name):
        self._result_file_name = file_name
        self._files_picker_result_label.config(text=path.basename(file_name))

    def __risks_set_result_file_name__(self, file_name):
        self._risks_result_file_name = file_name
        self._risks_files_picker_result_label.config(text=path.basename(file_name))

    def __reset_input_file__(self):
        self._input_file_name = None
        self._files_picker_input_label.config(text='select file')
        self.__load_data__()
        self.__update_info__()

    def __risks_reset_input_file__(self):
        self._risks_input_file_name = None
        self._risks_files_picker_input_label.config(text='select file')
        self.__risks_load_data__()

    def __reset_result_file__(self):
        self._result_file_name = None
        self._files_picker_result_label.config(text='select file')

    def __risks_reset_result_file__(self):
        self._risks_result_file_name = None
        self._risks_files_picker_result_label.config(text='select file')

    def __load_data__(self, data_dict=None):
        self._data = data_dict

    def __risks_load_data__(self, data_dict=None):
        self._risks_data = data_dict

    def __update_info__(self):
        if self._data is not None:
            x = self._data[0]
            x_dims = [len(i) for i in x]
            y_dim = len(self._data[1])
            self._vector_x1_dimension.config(text=str(x_dims[0]))
            self._vector_x2_dimension.config(text=str(x_dims[1]))
            self._vector_x3_dimension.config(text=str(x_dims[2]))
            self._vector_y_dimension.config(text=str(y_dim))
        else:
            self._vector_x1_dimension.config(text='')
            self._vector_x2_dimension.config(text='')
            self._vector_x3_dimension.config(text='')
            self._vector_y_dimension.config(text='')

    def __select_input_file__(self):
        picked_file = file_dialog.askopenfile()
        if picked_file is None:
            return None
        picked_file_name = picked_file.name
        try:
            data = __parse_file__(picked_file_name)
        except:
            self.__reset_input_file__()
            self.__show_error__('Open File Error', 'Cannot open file. Bad format')
            return
        self.__set_input_file_name__(picked_file_name)
        self.__load_data__(data)
        self.__update_info__()

    def __risks_select_input_file__(self):
        picked_file = file_dialog.askopenfile()
        if picked_file is None:
            return None
        picked_file_name = picked_file.name
        try:
            data = __risks_parse_file__(picked_file_name)
        except:
            self.__risks_reset_input_file__()
            self.__show_error__('Open File Error', 'Cannot open file. Bad format')
            return
        self.__risks_set_input_file_name__(picked_file_name)
        self.__risks_load_data__(data)

    def _make_plot(self):
        self._last_plots()

    def __del__(self):
        try:
            self._main_window.destroy()
        except:
            pass

    def reset_and_insert_results(self, results):
        self._result_window.config(state='normal')
        self._result_window.delete(1.0, 'end')
        self._result_window.insert(1.0, results)
        self._result_window.config(state='disabled')

    def __write_to_file__(self, data):
        open(self._result_file_name, 'w').write(data)

    def _switch_to_risks(self):
        self._main_window.withdraw()
        self._risks_window.deiconify()
        self.__reset_input_file__()
        self.__reset_result_file__()
        self.__load_data__()

    def _switch_to_main_window(self):
        self._risks_window.withdraw()
        self._main_window.deiconify()
        self.__risks_reset_input_file__()
        self.__risks_reset_result_file__()
        self.__risks_load_data__()

    def execute(self):
        self._main_window.mainloop()

    def set_size(self, height, width):
        self._main_window.geometry('{}x{}'.format(height, width))

    def resizeable(self, flag):
        self._main_window.resizable(width=flag, height=flag)


if __name__ == "__main__":
    a = Application()
    a.execute()
