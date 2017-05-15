#pragma once

#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <stdint.h> // <cstdint> requires c++11 support

#if __cplusplus > 199711L || _MSC_VER > 1800
#include <functional>
#endif

#include <Python.h>

#ifndef WITHOUT_NUMPY
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #include <numpy/arrayobject.h>
#endif // WITHOUT_NUMPY

#if PY_MAJOR_VERSION >= 3
#define PyString_FromString PyUnicode_FromString
#endif


namespace matplotlibcpp {

	namespace detail {
		struct _interpreter {
			PyObject *s_python_function_show;
			PyObject *s_python_function_save;
			PyObject *s_python_function_figure;
			PyObject *s_python_function_plot;
			PyObject *s_python_function_fill_between;
			PyObject *s_python_function_hist;
			PyObject *s_python_function_subplot;
			PyObject *s_python_function_legend;
			PyObject *s_python_function_xlim;
			PyObject *s_python_function_ylim;
			PyObject *s_python_function_title;
			PyObject *s_python_function_axis;
			PyObject *s_python_function_xlabel;
			PyObject *s_python_function_ylabel;
			PyObject *s_python_function_grid;
			PyObject *s_python_function_clf;
			PyObject *s_python_function_errorbar;
			PyObject *s_python_function_annotate;
			PyObject *s_python_function_tight_layout;
			PyObject *s_python_function_gca;
			PyObject *s_python_empty_tuple;

			// matplotlib.patches
			PyObject *s_python_class_patch;
			PyObject *s_python_class_ellipse;
			PyObject *s_python_class_arc;
			PyObject *s_python_class_arrow;
			PyObject *s_python_class_circle;
			PyObject *s_python_class_rectangle;
			PyObject *s_python_class_polygon;
			PyObject *s_python_class_fancyarrow;
			// and more



			/* For now, _interpreter is implemented as a singleton since its currently not possible to have
			   multiple independent embedded python interpreters without patching the python source code
			   or starting a separate process for each.
				http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
			   */

			static _interpreter& get() {
				static _interpreter ctx;
				return ctx;
			}

			private:
			_interpreter() {
                
                // optional but recommended
#if PY_MAJOR_VERSION >= 3
				wchar_t name[] = L"plotting";
#else
				char name[] = "plotting";
#endif
				Py_SetProgramName(name);
				Py_Initialize();

#ifndef WITHOUT_NUMPY
				import_array(); // initialize numpy C-API
#endif

				PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
				PyObject* pylabname  = PyString_FromString("pylab");
				PyObject* patchesname = PyString_FromString("matplotlib.patches");
				if(!pyplotname || !pylabname || !patchesname) { throw std::runtime_error("couldnt create string"); }

				PyObject* pymod = PyImport_Import(pyplotname);
				Py_DECREF(pyplotname);
				if(!pymod) { throw std::runtime_error("Error loading module matplotlib.pyplot!"); }

				PyObject* pylabmod = PyImport_Import(pylabname);
				Py_DECREF(pylabname);
				if(!pylabmod) { throw std::runtime_error("Error loading module pylab!"); }

				PyObject* patchesmod = PyImport_Import(patchesname);
				Py_DECREF(patchesname);
				if(!patchesmod) { throw std::runtime_error("Error loading module matplotlib.patches!"); }

				s_python_function_show = PyObject_GetAttrString(pymod, "show");
				s_python_function_figure = PyObject_GetAttrString(pymod, "figure");
				s_python_function_plot = PyObject_GetAttrString(pymod, "plot");
				s_python_function_fill_between = PyObject_GetAttrString(pymod, "fill_between");
				s_python_function_hist = PyObject_GetAttrString(pymod,"hist");
				s_python_function_subplot = PyObject_GetAttrString(pymod, "subplot");
				s_python_function_legend = PyObject_GetAttrString(pymod, "legend");
				s_python_function_ylim = PyObject_GetAttrString(pymod, "ylim");
				s_python_function_title = PyObject_GetAttrString(pymod, "title");
				s_python_function_axis = PyObject_GetAttrString(pymod, "axis");
				s_python_function_xlabel = PyObject_GetAttrString(pymod, "xlabel");
				s_python_function_ylabel = PyObject_GetAttrString(pymod, "ylabel");
				s_python_function_grid = PyObject_GetAttrString(pymod, "grid");
				s_python_function_xlim = PyObject_GetAttrString(pymod, "xlim");
				s_python_function_save = PyObject_GetAttrString(pylabmod, "savefig");
				s_python_function_annotate = PyObject_GetAttrString(pymod,"annotate");
				s_python_function_clf = PyObject_GetAttrString(pymod, "clf");
				s_python_function_errorbar = PyObject_GetAttrString(pymod, "errorbar");
				s_python_function_tight_layout = PyObject_GetAttrString(pymod, "tight_layout");
				s_python_function_gca = PyObject_GetAttrString(pymod, "gca");

				s_python_class_patch = PyObject_GetAttrString(patchesmod, "Patch");
				s_python_class_ellipse = PyObject_GetAttrString(patchesmod, "Ellipse");
				s_python_class_arc = PyObject_GetAttrString(patchesmod, "Arc");
				s_python_class_arrow = PyObject_GetAttrString(patchesmod, "Arrow");
				s_python_class_circle = PyObject_GetAttrString(patchesmod, "Circle");
				s_python_class_rectangle = PyObject_GetAttrString(patchesmod, "Rectangle");
				s_python_class_polygon = PyObject_GetAttrString(patchesmod, "Polygon");
				s_python_class_fancyarrow = PyObject_GetAttrString(patchesmod, "FancyArrow");

				if(        !s_python_function_show
					|| !s_python_function_figure
					|| !s_python_function_plot
					|| !s_python_function_fill_between
					|| !s_python_function_subplot
				   	|| !s_python_function_legend
					|| !s_python_function_ylim
					|| !s_python_function_title
					|| !s_python_function_axis
					|| !s_python_function_xlabel
					|| !s_python_function_ylabel
					|| !s_python_function_grid
					|| !s_python_function_xlim
					|| !s_python_function_save
					|| !s_python_function_clf
					|| !s_python_function_annotate
					|| !s_python_function_errorbar
					|| !s_python_function_errorbar
					|| !s_python_function_tight_layout
					|| !s_python_function_gca
					|| !s_python_class_patch
					|| !s_python_class_ellipse
					|| !s_python_class_arc
					|| !s_python_class_arrow
					|| !s_python_class_circle
					|| !s_python_class_rectangle
					|| !s_python_class_polygon
					|| !s_python_class_fancyarrow
				) { throw std::runtime_error("Couldn't find required function!"); }

				if (       !PyFunction_Check(s_python_function_show)
					|| !PyFunction_Check(s_python_function_figure)
					|| !PyFunction_Check(s_python_function_plot)
					|| !PyFunction_Check(s_python_function_fill_between)
					|| !PyFunction_Check(s_python_function_subplot)
					|| !PyFunction_Check(s_python_function_legend)
					|| !PyFunction_Check(s_python_function_annotate)
					|| !PyFunction_Check(s_python_function_ylim)
					|| !PyFunction_Check(s_python_function_title)
					|| !PyFunction_Check(s_python_function_axis)
					|| !PyFunction_Check(s_python_function_xlabel)
					|| !PyFunction_Check(s_python_function_ylabel)
					|| !PyFunction_Check(s_python_function_grid)
					|| !PyFunction_Check(s_python_function_xlim)
					|| !PyFunction_Check(s_python_function_save)
					|| !PyFunction_Check(s_python_function_clf)
					|| !PyFunction_Check(s_python_function_tight_layout)
					|| !PyFunction_Check(s_python_function_errorbar)
					|| !PyFunction_Check(s_python_function_gca)
				) { throw std::runtime_error("Python object is unexpectedly not a PyFunction."); }

				if (       !PyCallable_Check(s_python_class_patch)
					|| !PyCallable_Check(s_python_class_ellipse)
					|| !PyCallable_Check(s_python_class_arc)
					|| !PyCallable_Check(s_python_class_arrow)
					|| !PyCallable_Check(s_python_class_circle)
					|| !PyCallable_Check(s_python_class_rectangle)
					|| !PyCallable_Check(s_python_class_polygon)
					|| !PyCallable_Check(s_python_class_fancyarrow)
				) { throw std::runtime_error("Python object is unexpectedly not a PyClass."); }

				s_python_empty_tuple = PyTuple_New(0);
			}

			~_interpreter() {
				Py_Finalize();
			}
		};
	}
  
	bool annotate(std::string annotation, double x, double y)
	{
		PyObject * xy = PyTuple_New(2);
		PyObject * str = PyString_FromString(annotation.c_str());

		PyTuple_SetItem(xy,0,PyFloat_FromDouble(x));
		PyTuple_SetItem(xy,1,PyFloat_FromDouble(y));

		PyObject* kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "xy", xy);

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, str);

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_annotate, args, kwargs);
		
		Py_DECREF(args);
		Py_DECREF(kwargs);

		if(res) Py_DECREF(res);

		return res;
	}

#ifndef WITHOUT_NUMPY
	// Type selector for numpy array conversion
	template <typename T> struct select_npy_type { const static NPY_TYPES type = NPY_NOTYPE; }; //Default
	template <> struct select_npy_type<double> { const static NPY_TYPES type = NPY_DOUBLE; };
	template <> struct select_npy_type<float> { const static NPY_TYPES type = NPY_FLOAT; };
	template <> struct select_npy_type<bool> { const static NPY_TYPES type = NPY_BOOL; };
	template <> struct select_npy_type<int8_t> { const static NPY_TYPES type = NPY_INT8; };
	template <> struct select_npy_type<int16_t> { const static NPY_TYPES type = NPY_SHORT; };
	template <> struct select_npy_type<int32_t> { const static NPY_TYPES type = NPY_INT; };
	template <> struct select_npy_type<int64_t> { const static NPY_TYPES type = NPY_INT64; };
	template <> struct select_npy_type<uint8_t> { const static NPY_TYPES type = NPY_UINT8; };
	template <> struct select_npy_type<uint16_t> { const static NPY_TYPES type = NPY_USHORT; };
	template <> struct select_npy_type<uint32_t> { const static NPY_TYPES type = NPY_ULONG; };
	template <> struct select_npy_type<uint64_t> { const static NPY_TYPES type = NPY_UINT64; };

	template<typename Numeric>
	PyObject* get_array(const std::vector<Numeric>& v)
	{
		detail::_interpreter::get();	//interpreter needs to be initialized for the numpy commands to work
		NPY_TYPES type = select_npy_type<Numeric>::type; 
		if (type == NPY_NOTYPE)
		{
			std::vector<double> vd(v.size());
			npy_intp vsize = v.size();
			std::copy(v.begin(),v.end(),vd.begin());
			PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, NPY_DOUBLE, (void*)(vd.data()));
			return varray;
		}

		npy_intp vsize = v.size();
		PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, type, (void*)(v.data()));
		return varray;
	}

#else // fallback if we don't have numpy: copy every element of the given vector

	template<typename Numeric>
	PyObject* get_array(const std::vector<Numeric>& v)
	{
		PyObject* list = PyList_New(v.size());
		for(size_t i = 0; i < v.size(); ++i) {
			PyList_SetItem(list, i, PyFloat_FromDouble(v.at(i)));
		}
		return list;
	}

#endif // WITHOUT_NUMPY

	template<typename Numeric>
	bool plot(const std::vector<Numeric> &x, const std::vector<Numeric> &y, const std::map<std::string, std::string>& keywords)
	{
		assert(x.size() == y.size());

		// using numpy arrays
		PyObject* xarray = get_array(x);
		PyObject* yarray = get_array(y);

		// construct positional args
		PyObject* args = PyTuple_New(2);
		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, yarray);

		// construct keyword args
		PyObject* kwargs = PyDict_New();
		for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
		{
			PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
		}

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, args, kwargs);

		Py_DECREF(args);
		Py_DECREF(kwargs);
		if(res) Py_DECREF(res);

		return res;
	}

	template< typename Numeric >
	bool fill_between(const std::vector<Numeric>& x, const std::vector<Numeric>& y1, const std::vector<Numeric>& y2, const std::map<std::string, std::string>& keywords)
	{
		assert(x.size() == y1.size());
		assert(x.size() == y2.size());

		// using numpy arrays
		PyObject* xarray = get_array(x);
		PyObject* y1array = get_array(y1);
		PyObject* y2array = get_array(y2);

		// construct positional args
		PyObject* args = PyTuple_New(3);
		PyTuple_SetItem(args, 0, xarray);
		PyTuple_SetItem(args, 1, y1array);
		PyTuple_SetItem(args, 2, y2array);

		// construct keyword args
		PyObject* kwargs = PyDict_New();
		for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
		{
			PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
		}

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_fill_between, args, kwargs);

		Py_DECREF(args);
		Py_DECREF(kwargs);
		if(res) Py_DECREF(res);

		return res;
	}

	template< typename Numeric>
	bool hist(const std::vector<Numeric>& y, long bins=10,std::string color="b", double alpha=1.0)
	{

		PyObject* yarray = get_array(y);

		PyObject* kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
		PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
		PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
		

		PyObject* plot_args = PyTuple_New(1);

		PyTuple_SetItem(plot_args, 0, yarray);


		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);


		Py_DECREF(plot_args);
		Py_DECREF(kwargs);
		if(res) Py_DECREF(res);

		return res;
	}

	template< typename Numeric>
	bool named_hist(std::string label,const std::vector<Numeric>& y, long bins=10, std::string color="b", double alpha=1.0)
	{
		PyObject* yarray = get_array(y);

		PyObject* kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
		PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
		PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
		PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));


		PyObject* plot_args = PyTuple_New(1);
		PyTuple_SetItem(plot_args, 0, yarray);

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_hist, plot_args, kwargs);

		Py_DECREF(plot_args);
		Py_DECREF(kwargs);
		if(res) Py_DECREF(res);

		return res;
	}
	
	template<typename NumericX, typename NumericY>
	bool plot(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
	{
		assert(x.size() == y.size());

		PyObject* xarray = get_array(x);
		PyObject* yarray = get_array(y);

		PyObject* pystring = PyString_FromString(s.c_str());

		PyObject* plot_args = PyTuple_New(3);
		PyTuple_SetItem(plot_args, 0, xarray);
		PyTuple_SetItem(plot_args, 1, yarray);
		PyTuple_SetItem(plot_args, 2, pystring);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, plot_args);

		Py_DECREF(plot_args);
		if(res) Py_DECREF(res);

		return res;
	}

	template<typename NumericX, typename NumericY>
	bool errorbar(const std::vector<NumericX> &x, const std::vector<NumericY> &y, const std::vector<NumericX> &yerr, const std::string &s = "")
	{
		assert(x.size() == y.size());

		PyObject* xarray = get_array(x);
		PyObject* yarray = get_array(y);
		PyObject* yerrarray = get_array(yerr);

		PyObject *kwargs = PyDict_New();

		PyDict_SetItemString(kwargs, "yerr", yerrarray);

		PyObject *pystring = PyString_FromString(s.c_str());

		PyObject *plot_args = PyTuple_New(2);
		PyTuple_SetItem(plot_args, 0, xarray);
		PyTuple_SetItem(plot_args, 1, yarray);

		PyObject *res = PyObject_Call(detail::_interpreter::get().s_python_function_errorbar, plot_args, kwargs);

		Py_DECREF(kwargs);
		Py_DECREF(plot_args);

		if (res)
			Py_DECREF(res);
		else
			throw std::runtime_error("Call to errorbar() failed.");

		return res;
	}

	template<typename Numeric>
	bool named_plot(const std::string& name, const std::vector<Numeric>& y, const std::string& format = "")
	{
		PyObject* kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

		PyObject* yarray = get_array(y);

		PyObject* pystring = PyString_FromString(format.c_str());

		PyObject* plot_args = PyTuple_New(2);

		PyTuple_SetItem(plot_args, 0, yarray);
		PyTuple_SetItem(plot_args, 1, pystring);

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);

		Py_DECREF(kwargs);
		Py_DECREF(plot_args);
		if(res) Py_DECREF(res);

		return res;
	}

	template<typename Numeric>
	bool named_plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
	{
		PyObject* kwargs = PyDict_New();
		PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

		PyObject* xarray = get_array(x);
		PyObject* yarray = get_array(y);

		PyObject* pystring = PyString_FromString(format.c_str());

		PyObject* plot_args = PyTuple_New(3);
		PyTuple_SetItem(plot_args, 0, xarray);
		PyTuple_SetItem(plot_args, 1, yarray);
		PyTuple_SetItem(plot_args, 2, pystring);

		PyObject* res = PyObject_Call(detail::_interpreter::get().s_python_function_plot, plot_args, kwargs);

		Py_DECREF(kwargs);
		Py_DECREF(plot_args);
		if(res) Py_DECREF(res);

		return res;
	}

	template<typename Numeric>
	bool plot(const std::vector<Numeric>& y, const std::string& format = "")
	{
		std::vector<Numeric> x(y.size());
		for(size_t i=0; i<x.size(); ++i) x.at(i) = i;
		return plot(x,y,format);
	}

	inline void figure()
	{
		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_figure, detail::_interpreter::get().s_python_empty_tuple);
		if(!res) throw std::runtime_error("Call to figure() failed.");

		Py_DECREF(res);
	}

	inline void legend()
	{
		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_legend, detail::_interpreter::get().s_python_empty_tuple);
		if(!res) throw std::runtime_error("Call to legend() failed.");

		Py_DECREF(res);
	}

	template<typename Numeric>
	void ylim(Numeric left, Numeric right)
	{
		PyObject* list = PyList_New(2);
		PyList_SetItem(list, 0, PyFloat_FromDouble(left));
		PyList_SetItem(list, 1, PyFloat_FromDouble(right));

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, list);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylim, args);
		if(!res) throw std::runtime_error("Call to ylim() failed.");

		Py_DECREF(args);
		Py_DECREF(res);
	}

	template<typename Numeric>
	void xlim(Numeric left, Numeric right)
	{
		PyObject* list = PyList_New(2);
		PyList_SetItem(list, 0, PyFloat_FromDouble(left));
		PyList_SetItem(list, 1, PyFloat_FromDouble(right));

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, list);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlim, args);
		if(!res) throw std::runtime_error("Call to xlim() failed.");

		Py_DECREF(args);
		Py_DECREF(res);
	}
  
  
	inline double* xlim()
	{
		PyObject* args = PyTuple_New(0);
		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlim, args);
		PyObject* left = PyTuple_GetItem(res,0);
		PyObject* right = PyTuple_GetItem(res,1);

		double* arr = new double[2];
		arr[0] = PyFloat_AsDouble(left);
		arr[1] = PyFloat_AsDouble(right);
    
		if(!res) throw std::runtime_error("Call to xlim() failed.");

		Py_DECREF(res);
		return arr;
	}
  
  
	inline double* ylim()
	{
		PyObject* args = PyTuple_New(0);
		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylim, args);
		PyObject* left = PyTuple_GetItem(res,0);
		PyObject* right = PyTuple_GetItem(res,1);

		double* arr = new double[2];
		arr[0] = PyFloat_AsDouble(left);
		arr[1] = PyFloat_AsDouble(right);
    
		if(!res) throw std::runtime_error("Call to ylim() failed."); 

		Py_DECREF(res);
		return arr;
	}

	inline void subplot(long nrows, long ncols, long plot_number)
	{
		// construct positional args
		PyObject* args = PyTuple_New(3);
		PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
		PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
		PyTuple_SetItem(args, 2, PyFloat_FromDouble(plot_number));

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_subplot, args);
		if(!res) throw std::runtime_error("Call to subplot() failed.");

		Py_DECREF(args);
		Py_DECREF(res);
	}

	inline void title(const std::string &titlestr)
	{
		PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, pytitlestr);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_title, args);
		if(!res) throw std::runtime_error("Call to title() failed.");

		// if PyDeCRFF, the function doesn't work on Mac OS
	}

	inline void axis(const std::string &axisstr)
	{
		PyObject* str = PyString_FromString(axisstr.c_str());
		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, str);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_axis, args);
		if(!res) throw std::runtime_error("Call to title() failed.");

		// if PyDeCRFF, the function doesn't work on Mac OS
	}

	inline void xlabel(const std::string &str)
	{
		PyObject* pystr = PyString_FromString(str.c_str());
		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, pystr);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_xlabel, args);
		if(!res) throw std::runtime_error("Call to xlabel() failed.");

		// if PyDeCRFF, the function doesn't work on Mac OS
	}

	inline void ylabel(const std::string &str)
	{
		PyObject* pystr = PyString_FromString(str.c_str());
		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, pystr);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_ylabel, args);
		if(!res) throw std::runtime_error("Call to ylabel() failed.");

		// if PyDeCRFF, the function doesn't work on Mac OS
	}

	inline void grid(bool flag)
	{
		PyObject* pyflag = flag ? Py_True : Py_False;

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, pyflag);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_grid, args);
		if(!res) throw std::runtime_error("Call to grid() failed.");

		// if PyDeCRFF, the function doesn't work on Mac OS
	}

	inline void show()
	{
		PyObject* res = PyObject_CallObject(
			detail::_interpreter::get().s_python_function_show,
			detail::_interpreter::get().s_python_empty_tuple);

		if (!res) throw std::runtime_error("Call to show() failed.");

		Py_DECREF(res);
	}

	inline void save(const std::string& filename)
	{
		PyObject* pyfilename = PyString_FromString(filename.c_str());

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, pyfilename);

		PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_save, args);
		if (!res) throw std::runtime_error("Call to save() failed.");

		Py_DECREF(args);
		Py_DECREF(res);
	}

	inline void clf() {
		PyObject *res = PyObject_CallObject(
			detail::_interpreter::get().s_python_function_clf,
			detail::_interpreter::get().s_python_empty_tuple);

		if (!res) throw std::runtime_error("Call to clf() failed.");

		Py_DECREF(res);
	}

	// Actually, is there any reason not to call this automatically for every plot?
	inline void tight_layout() {
		PyObject *res = PyObject_CallObject(
			detail::_interpreter::get().s_python_function_tight_layout,
			detail::_interpreter::get().s_python_empty_tuple);

		if (!res) throw std::runtime_error("Call to tight_layout() failed.");

		Py_DECREF(res);
	}

#if __cplusplus > 199711L || _MSC_VER > 1800
	// C++11-exclusive content starts here (variadic plot() and initializer list support)

	namespace detail {
		template<typename T>
		using is_function = typename std::is_function<std::remove_pointer<std::remove_reference<T>>>::type;

		template<bool obj, typename T>
		struct is_callable_impl;

		template<typename T>
		struct is_callable_impl<false, T>
		{
			typedef is_function<T> type;
		}; // a non-object is callable iff it is a function

		template<typename T>
		struct is_callable_impl<true, T>
		{
			struct Fallback { void operator()(); };
			struct Derived : T, Fallback { };

			template<typename U, U> struct Check;

			template<typename U>
			static std::true_type test( ... ); // use a variadic function to make sure (1) it accepts everything and (2) its always the worst match

			template<typename U>
			static std::false_type test( Check<void(Fallback::*)(), &U::operator()>* );

		public:
			typedef decltype(test<Derived>(nullptr)) type;
			typedef decltype(&Fallback::operator()) dtype;
			static constexpr bool value = type::value;
		}; // an object is callable iff it defines operator()

		template<typename T>
		struct is_callable
		{
			// dispatch to is_callable_impl<true, T> or is_callable_impl<false, T> depending on whether T is of class type or not
			typedef typename is_callable_impl<std::is_class<T>::value, T>::type type;
		};

		template<typename IsYDataCallable>
		struct plot_impl { };

		template<>
		struct plot_impl<std::false_type>
		{
			template<typename IterableX, typename IterableY>
			bool operator()(const IterableX& x, const IterableY& y, const std::string& format)
			{
				// 2-phase lookup for distance, begin, end
				using std::distance;
				using std::begin;
				using std::end;

				auto xs = distance(begin(x), end(x));
				auto ys = distance(begin(y), end(y));
				assert(xs == ys && "x and y data must have the same number of elements!");

				PyObject* xlist = PyList_New(xs);
				PyObject* ylist = PyList_New(ys);
				PyObject* pystring = PyString_FromString(format.c_str());

				auto itx = begin(x), ity = begin(y);
				for(size_t i = 0; i < xs; ++i) {
					PyList_SetItem(xlist, i, PyFloat_FromDouble(*itx++));
					PyList_SetItem(ylist, i, PyFloat_FromDouble(*ity++));
				}

				PyObject* plot_args = PyTuple_New(3);
				PyTuple_SetItem(plot_args, 0, xlist);
				PyTuple_SetItem(plot_args, 1, ylist);
				PyTuple_SetItem(plot_args, 2, pystring);

				PyObject* res = PyObject_CallObject(detail::_interpreter::get().s_python_function_plot, plot_args);

				Py_DECREF(plot_args);
				if(res) Py_DECREF(res);

				return res;
			}
		};

		template<>
		struct plot_impl<std::true_type>
		{
			template<typename Iterable, typename Callable>
			bool operator()(const Iterable& ticks, const Callable& f, const std::string& format)
			{
				//std::cout << "Callable impl called" << std::endl;

				if(begin(ticks) == end(ticks)) return true;

				// We could use additional meta-programming to deduce the correct element type of y,
				// but all values have to be convertible to double anyways
				std::vector<double> y;
				for(auto x : ticks) y.push_back(f(x));
				return plot_impl<std::false_type>()(ticks,y,format);
			}
		};
	}

	// recursion stop for the above
	template<typename... Args>
	bool plot() { return true; }

	template<typename A, typename B, typename... Args>
	bool plot(const A& a, const B& b, const std::string& format, Args... args)
	{
		return detail::plot_impl<typename detail::is_callable<B>::type>()(a,b,format) && plot(args...);
	}

	/*
	 * This group of plot() functions is needed to support initializer lists, i.e. calling
	 *    plot( {1,2,3,4} )
	 */
	bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
		return plot<double,double>(x,y,format);
	}

	bool plot(const std::vector<double>& y, const std::string& format = "") {
		return plot<double>(y,format);
	}

	bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::map<std::string, std::string>& keywords) {
		return plot<double>(x,y,keywords);
	}

	bool named_plot(const std::string& name, const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
		return named_plot<double>(name,x,y,format);
	}

#endif

	namespace patches {

		// TODO move somewhere else
		class BaseWrapper
		{
		public:
			BaseWrapper() : _impl(nullptr) {}
			BaseWrapper(PyObject *impl) : _impl(impl)
			{
				if (_impl)
					Py_INCREF(_impl);
			}
			BaseWrapper(const BaseWrapper &other) : _impl(other._impl)
			{
				if (_impl)
					Py_INCREF(_impl);
			}
			BaseWrapper(BaseWrapper &&other) : _impl(other._impl)
			{
				other._impl = nullptr;
			}

			// Non-virtual descructor (is this enough?)
			~BaseWrapper()
			{
				if (_impl)
					Py_DECREF(_impl);
			}

			operator PyObject *() const
			{
				return _impl;
			}

		protected:
			PyObject *_impl;
		};

		class Patch : public BaseWrapper
		{
		public:
			// TODO
		protected:
			Patch() {}
		};

		class Ellipse : public Patch
		{
		public:
			Ellipse(double x, double y, double width, double height, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Ellipse(x, y, width, height, 0.0, keywords)
			{}

			Ellipse(double x, double y, double width, double height, double angle, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				PyObject* xy = PyTuple_New(2);
				PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));

				// construct positional args
				PyObject* args = PyTuple_New(4);
				PyTuple_SetItem(args, 0, xy);
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(width));
				PyTuple_SetItem(args, 2, PyFloat_FromDouble(height));
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(angle));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
				}

				_impl = PyObject_Call(::matplotlibcpp::detail::_interpreter::get().s_python_class_ellipse, args, kwargs);

				Py_DECREF(xy);
				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			Ellipse() {}
		};

		class Circle : public Ellipse
		{
		public:
			Circle(double x, double y, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Circle(x, y, 5.0, keywords)
			{}

			Circle(double x, double y, double radius, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				PyObject* xy = PyTuple_New(2);
				PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));

				// construct positional args
				PyObject* args = PyTuple_New(2);
				PyTuple_SetItem(args, 0, xy);
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(radius));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
				}

				_impl = PyObject_Call(::matplotlibcpp::detail::_interpreter::get().s_python_class_circle, args, kwargs);

				Py_DECREF(xy);
				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			Circle() {}
		};

		class Arc : public Ellipse
		{
		public:
			Arc(double x, double y, double width, double height, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Arc(x, y, width, height, 0.0, 0.0, 360.0, keywords)
			{}
			Arc(double x, double y, double width, double height, double angle, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Arc(x, y, width, height, angle, 0.0, 360.0, keywords)
			{}
			Arc(double x, double y, double width, double height, double angle, double theta1, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Arc(x, y, width, height, angle, theta1, 360.0, keywords)
			{}

			Arc(double x, double y, double width, double height, double angle, double theta1, double theta2, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				PyObject* xy = PyTuple_New(2);
				PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));

				// construct positional args
				PyObject* args = PyTuple_New(6);
				PyTuple_SetItem(args, 0, xy);
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(width));
				PyTuple_SetItem(args, 2, PyFloat_FromDouble(height));
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(angle));
				PyTuple_SetItem(args, 4, PyFloat_FromDouble(theta1));
				PyTuple_SetItem(args, 5, PyFloat_FromDouble(theta2));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
				}

				_impl = PyObject_Call(::matplotlibcpp::detail::_interpreter::get().s_python_class_arc, args, kwargs);

				Py_DECREF(xy);
				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			Arc() {}
		};

		class Arrow : public Patch
		{
		public:
			Arrow(double x, double y, double dx, double dy, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Arrow(x, y, dx, dy, 1.0, keywords)
			{}

			Arrow(double x, double y, double dx, double dy, double width, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				// construct positional args
				PyObject* args = PyTuple_New(5);
				PyTuple_SetItem(args, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(y));
				PyTuple_SetItem(args, 2, PyFloat_FromDouble(dx));
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(dy));
				PyTuple_SetItem(args, 4, PyFloat_FromDouble(width));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
				}

				_impl = PyObject_Call(::matplotlibcpp::detail::_interpreter::get().s_python_class_arrow, args, kwargs);

				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			Arrow() {}
		};

		class Rectangle : public Patch
		{
		public:
			Rectangle(double x, double y, double width, double height, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					Rectangle(x, y, width, height, 0.0, keywords)
			{}

			Rectangle(double x, double y, double width, double height, double angle, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				PyObject* xy = PyTuple_New(2);
				PyTuple_SetItem(xy, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(xy, 1, PyFloat_FromDouble(y));

				// construct positional args
				PyObject* args = PyTuple_New(4);
				PyTuple_SetItem(args, 0, xy);
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(width));
				PyTuple_SetItem(args, 2, PyFloat_FromDouble(height));
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(angle));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
				}

				_impl = PyObject_Call(::matplotlibcpp::detail::_interpreter::get().s_python_class_rectangle, args, kwargs);

				Py_DECREF(xy);
				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			Rectangle() {}
		};

		class Polygon : public Patch
		{
		public:
			// TODO requires numpy array...

		protected:
			Polygon() {}
		};

		class FancyArrow : public Polygon
		{
		public:
			FancyArrow(double x, double y, double dx, double dy, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>()) :
					FancyArrow(x, y, dx, dy, 0.001)
			{}

			FancyArrow(double x, double y, double dx, double dy, double width, const std::map<std::string, std::string> &keywords = std::map<std::string, std::string>())
			{
				// construct positional args
				PyObject* args = PyTuple_New(11);
				PyTuple_SetItem(args, 0, PyFloat_FromDouble(x));
				PyTuple_SetItem(args, 1, PyFloat_FromDouble(y));
				PyTuple_SetItem(args, 2, PyFloat_FromDouble(dx));
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(dy));
				PyTuple_SetItem(args, 4, PyFloat_FromDouble(width));

				// default values
				PyTuple_SetItem(args, 5, PyBool_FromLong(0));
				PyTuple_SetItem(args, 6, Py_None);
				PyTuple_SetItem(args, 7, Py_None);
				PyTuple_SetItem(args, 8, PyString_FromString("full"));
				PyTuple_SetItem(args, 9, PyFloat_FromDouble(0));
				PyTuple_SetItem(args, 10, PyBool_FromLong(0));

				// construct keyword args
				PyObject* kwargs = PyDict_New();
				for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
				{
					if (it->first == "length_includes_head")
					{
						PyTuple_SetItem(args, 5, PyBool_FromLong(std::stol(it->second)));
					}
					else if (it->first == "head_width")
					{
						PyTuple_SetItem(args, 6, PyFloat_FromDouble(std::stod(it->second)));
					}
					else if (it->first == "head_length")
					{
						PyTuple_SetItem(args, 7, PyFloat_FromDouble(std::stod(it->second)));
					}
					else if (it->first == "shape")
					{
						PyTuple_SetItem(args, 8, PyString_FromString(it->second.c_str()));
					}
					else if (it->first == "overhang")
					{
						PyTuple_SetItem(args, 9, PyFloat_FromDouble(std::stod(it->second)));
					}
					else if (it->first == "head_starts_at_zero")
					{
						PyTuple_SetItem(args, 10, PyBool_FromLong(std::stol(it->second)));
					}
					else
					{
						PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
					}
				}

				_impl = PyObject_Call(detail::_interpreter::get().s_python_class_fancyarrow, args, kwargs);

				Py_DECREF(args);
				Py_DECREF(kwargs);
			}

		protected:
			FancyArrow() {}
		};

	} // namespace patches

	void add_patch(const patches::Patch &patch)
	{
		// This method is actually a member of Axes, so apply this to the global current axes...
		PyObject* axes = PyObject_CallObject(detail::_interpreter::get().s_python_function_gca, nullptr);
		PyObject* add_patch_name = PyString_FromString("add_patch"); // TODO static
		// if (!PyInstance_Check_Check((PyObject*)patch)) throw std::runtime_error("invalid patch");
		PyObject* res = PyObject_CallMethodObjArgs(axes, add_patch_name, (PyObject*)patch, nullptr);

		if (!res) throw std::runtime_error("Call to add_patch() failed.");

		Py_DECREF(axes);
		Py_DECREF(add_patch_name);
		Py_DECREF(res);
	}

}
