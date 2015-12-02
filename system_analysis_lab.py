#!/usr/bin/python

import tkinter as tk
import tkinter.filedialog as file_dialog
import tkinter.messagebox as message_box
from collections import defaultdict
from os import path
import pandas
from constants import *
from functional_restoration.multiplicative import find_best_degrees as fbd_mul, make_model as calc_mul
from functional_restoration.model.additive import Additive, AdditiveDegreeFinder

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
    data_dict = {'x': {}, 'y': {}}
    tmp_dict = defaultdict(list)
    for column in sorted(columns):
        tmp_dict[column[:2].lower()].append(data[column].tolist())

    for i in sorted(tmp_dict.keys()):
        if i[0] == 'x':
            data_dict['x'][i] = tmp_dict[i]
        elif i[0] == 'y':
            data_dict['y'][i] = tmp_dict[i][0]

    if len(data_dict['x']) > 3:
        print("sry, it's only experimental program, you can not have more than 3 x-variables")
        raise Exception

    return data_dict


class Application:
    @staticmethod
    def __show_error__(title, message):
        message_box.showerror(title, message)

    def __init__(self, root=tk.Tk()):
        self._main_window = root
        self._main_window.title('System analysis')
        self._validator = self._main_window.register(__validate_only_digits__)
        self.__init_widgets__()
        # self.set_size(600,480)
        # self.resizeable(False)

    def __init_widgets__(self):
        self._last_plots = None

        self._data = None

        self._last_result = None

        self._samples = tk.StringVar()
        self._samples.set('50')

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

        self._plot_mode = tk.IntVar()
        self._plot_mode.set(1)

        self._top_block = tk.Frame(self._main_window)
        self._middle_block = tk.Frame(self._main_window)
        self._bottom_block = tk.Frame(self._main_window)

        self._input_file_name = None
        self._result_file_name = None

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

    def __set_input_file_name__(self, file_name):
        self._input_file_name = file_name
        self._files_picker_input_label.config(text=path.basename(file_name))

    def __reset_input_file__(self):
        self._input_file_name = None
        self._files_picker_input_label.config(text='select file')
        self.__reset_data__()
        self.__update_info__()

    def __set_result_file_name__(self, file_name):
        self._result_file_name = file_name
        self._files_picker_result_label.config(text=path.basename(file_name))

    def __load_data__(self, data_dict):
        self._data = data_dict

    def __reset_data__(self):
        self._data = None

    def __update_info__(self):
        if self._data is not None:
            x = self._data['x']
            x_dims = [len(x[i]) for i in sorted(x)]
            y_dim = len(self._data['y'])
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

        if form == 'mul':
            if find_best_degrees:
                results, self._last_plots = fbd_mul(self._data, degrees, weights, method, polynom, find_lambda,
                                                    epsilon=eps)
            else:
                results, self._last_plots = calc_mul(self._data, degrees, weights, method, polynom, find_lambda,
                                                     epsilon=eps)
        else:
            if find_best_degrees:
                model = AdditiveDegreeFinder(degrees, weights, method, polynom, find_lambda)
                res = model.fit(self._data)
                results = res.text()
                self._last_plots = res.plot
            else:
                model = Additive(degrees, weights, method, polynom, find_lambda)
                res = model.fit(self._data)
                results = res.text()
                self._last_plots = res.plot

        self.reset_and_insert_results(results)
        self.__write_to_file__(results)

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

    def execute(self):
        self._main_window.mainloop()

    def set_size(self, height, width):
        self._main_window.geometry('{}x{}'.format(height, width))

    def resizeable(self, flag):
        self._main_window.resizable(width=flag, height=flag)


if __name__ == "__main__":
    a = Application()
    a.execute()
