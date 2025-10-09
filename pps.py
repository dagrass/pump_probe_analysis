# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:25:04 2023

@author: david
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mlp
from skimage import io
import re


class PPS:

    def __init__(self, data, dataType="data", filename="unkown", mask=None):
        """
        Import pump-probe stack.

        Parameters
        ----------
        data : str or [images, delays]
            If str must be a filename of DukeScan, mathematica, or pickle pps
            file. Otherwise data in form of [images, delays].
        dataType : str, optional
            Either mathematica, DukeScan, pickle, or data. The default is
            "data".
        filename : str, optional
            If dataType=data, this string is saved as the filename of the pps
            instance. The default is "unkown".
        mask : np.bool_, optional
            This array is set as mask of this pps instance. The default is
            None.

        Returns
        -------
        None.

        """
        if dataType == "mathematica":
            if isinstance(data, str):
                temp = PPS.import_stack_mathematica(data)
                self.images = np.array(temp[0], dtype=np.float64)
                self.times = np.array(temp[1])
                self.filename = data
                self.image_dimensions = self.images[0].shape
                # check if there are 0 value pixel and derive mask
                # becasue there is no mask array yet we can obviously not use
                # it to create the mask here:
                self.mask = self.project(maskOn=False) != 0

        elif dataType == "DukeScan":
            self.images = np.array(io.imread_collection(data)[0], dtype=np.float64)
            self.times = PPS.time_delays(data)
            self.filename = data
            self.image_dimensions = self.images[0].shape
            # check if there are 0 value pixel and derive mask
            # becasue there is no mask array yet we can obviously not use it to
            # create the mask here:
            self.mask = self.project(maskOn=False) != 0

        elif dataType == "pickle":
            import pickle

            with open(data, "rb") as f:
                save_object = pickle.load(f)

            self.images = save_object["images"]
            self.times = save_object["times"]
            self.filename = save_object["filename"]
            self.image_dimensions = save_object["image_dimensions"]
            self.mask = save_object["mask"]

        elif dataType == "data":
            self.images = np.array(data[0], dtype=np.float64)
            self.times = np.array(data[1])
            self.filename = filename
            self.image_dimensions = self.images[0].shape

            # deal with mask
            if mask is None:
                # check if there are 0 value pixel and derive mask
                # becasue there is no mask array yet we can obviously not use
                # it to create the mask here:
                self.mask = self.project(maskOn=False) != 0

            else:
                self.mask = mask

        else:
            print("PPS constructor failed")

        self.results = {}

    def save(self, filename):
        """
        Save stack (PPS instance) into filename.

        Parameters
        ----------
        filename : str
            Filename of saved stack.

        Returns
        -------
        None.

        """
        import pickle

        save_object = {
            "images": self.images,
            "times": self.times,
            "filename": self.filename,
            "image_dimensions": self.image_dimensions,
            "mask": self.mask,
        }

        with open(filename, "wb") as f:
            pickle.dump(save_object, f)

    @staticmethod
    def time_delays(fn):
        """
        Import time delaus from log file or x-axis file (older stacks).

        Parameters
        ----------
        fn : str
            Filename of pp stack (Tif file).

        Returns
        -------
        times : np array of float
            Time delays.

        """
        # check if (older) x-axis file still exists
        fn_new = fn.replace(".tif", "_xaxis.txt")
        if os.path.isfile(fn_new):
            import pandas as pd

            times = pd.read_table(fn_new, header=None).iloc[:, 0].to_numpy()
            return times

        # import conventional log file
        if fn.endswith("_DS_CH1.tif"):
            fn_new = fn.replace("_DS_CH1.tif", ".log")
        elif fn.endswith("_DS_CH2.tif"):
            fn_new = fn.replace("_DS_CH2.tif", ".log")
        elif fn.endswith("_DS_CH3.tif"):
            fn_new = fn.replace("_DS_CH3.tif", ".log")
        elif fn.endswith("_DS_CH4.tif"):
            fn_new = fn.replace("_DS_CH4.tif", ".log")
        try:
            with open(fn_new, "r", encoding="utf-8") as f:
                log = f.read()
        except UnicodeDecodeError:
            with open(fn_new, "r", encoding="latin1") as f:
                log = f.read()

        fn_new = fn.replace("_DS_CH1.tif", ".log")

        with open(fn_new, "r") as f:
            log = f.read()
        match = re.search(r"(?:delayArr_ps = )([\-0-9,.]+).*", log)
        if match:
            times = np.fromstring(match.group(1), dtype=float, sep=",")
        else:
            print("there was a problem importing time delays with", fn)
            times = []
        return times

    @staticmethod
    def _substack_index_1d(size_image, size_sub):
        # comput number of sub images along 1-d
        n_sub_images = int(np.ceil(size_image / size_sub))

        index = []
        for i in range(n_sub_images):
            if (i + 1) * size_sub < size_image:
                index.append([i * size_sub, (i + 1) * size_sub])
            else:
                index.append([i * size_sub, size_image])
        return np.array(index)

    def _substack_index(self, size):
        index_x = PPS._substack_index_1d(self.image_dimensions[0], size)
        index_y = PPS._substack_index_1d(self.image_dimensions[1], size)
        return [index_x, index_y]

    @staticmethod
    def import_stack_mathematica(filename):
        """Load stack saved in david's mathematica format.

        filename is the name of the file that should be imported
        function returns a stack in david's mathematica convention
        stack = [[img(t=t0), img(t=t1), ...], timeAxis]
        """
        # open file as read-binary with no buffering
        f = open(filename, "rb", buffering=0)

        # import first three int16 which are the time, x and y dimension
        # convert dim to int64, because int16 is not big enougth for
        # multiplications
        dim = np.fromfile(f, dtype=np.int16, count=3)
        dim = dim.astype(np.int64)

        # import the time axis
        time = np.fromfile(f, dtype=np.float64, count=dim[0])

        # import stack, loop over time dimension dim[0] and reshape to image
        # dimensions
        images = []
        for i in range(dim[0]):
            temp = np.fromfile(f, dtype=np.float64, count=dim[1] * dim[2])
            images.append(temp.reshape((dim[1], dim[2])))

        # close file
        f.close

        return [images, time]

    def average_times(self, time_averages):
        # find indicies of time delays to be averaged -> positions
        positions = []
        for j in time_averages:
            temp_j = []
            for i in j:
                temp = np.argwhere(np.array(self.times) == i)[0, 0]
                temp_j.append(temp)
            positions.append(temp_j)

        # compute avegrae time delays and images
        new_times = []
        new_images = []
        for i in positions:
            new_times.append(np.mean(np.array(self.times)[i]))
            new_images.append(np.mean(self.images[i], axis=0))

        # redefine time delays and image stacks
        self.times = new_times
        self.images = np.array(new_images)

    def select_delays(self, delays="melanoma1"):
        """
        Select time delays of image stack.

        Parameters----------
        delays : TYPE, str or list of float
            DESCRIPTION. The default is "melanoma1".
            "melanoma1" = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.25, 0.5, 0.6, 0.7,
                           0.8, 0.9, 1.0, 2.5, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 29.0, 40.0,
                           50.0, 60.0, 70.0, 80.0]
            otherwise it can just be alist of time delays
        Returns
        -------
        None.

        """
        # check if delays are standard delays for machine learning melanoma
        # attempts
        if (type(delays) is str) and (delays == "melanoma1"):
            delays = [
                -1.0,
                -0.5,
                -0.1,
                0.0,
                0.1,
                0.25,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                2.5,
                2.0,
                3.0,
                3.5,
                4.0,
                4.5,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                15.0,
                20.0,
                29.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ]
        # find array indicies of delays
        pos = []
        for i in delays:
            pos.append(np.where(self.times == i)[0][0])

        # redefine time delays and image stacks
        self.times = self.times[pos]
        self.images = self.images[pos]

    def subtractFirst(self, n=1):
        """
        Subtract (average) of first n images from stack.

        Parameters
        ----------
        n : int, optional
            Number of stacks to be averaged and subtracted. The default is 1.

        Returns
        -------
        None.

        """
        mean = np.mean(self.images[:n], axis=0)
        self.images = self.images - mean

    def normalize(self, norm="minmax", inPlace=False):
        """
        Normalize pp stack.

        Parameters
        ----------
        norm : str, optional
            Norm to be used. The default is "minmax".
        inPlace : TYPE, optional
            If True this instance will be normalized. The default is False.

        Returns
        -------
        pp stack
            Returns normalized stack.

        """
        if norm is None:
            pass
        elif norm == "minmax":
            avg = self.avg()
            extremum = np.max(np.abs([np.min(avg), np.max(avg)]))

        if inPlace:
            self.images = self.images / extremum

        return PPS([self.images / extremum, self.times], mask=self.mask)

    def avg(self, maskOn=True, norm=None):
        """
        Compute average TA curve of pp stack.

        Parameters
        ----------
        maskOn : Boolean, optional
            Use mask for computation. The default is True.
        norm : str, optional
            Use norm for avg computation, i.e. minmax. The default is None.

        Returns
        -------
        avg : np array of float
            Averaged TA curve.

        """
        if maskOn is True:
            avg = [np.mean(i[self.mask]) for i in self.images]
        else:
            avg = [np.mean(i) for i in self.images]

        if norm is None:
            pass
        elif norm == "minmax":
            extremum = np.max(np.abs([np.min(avg), np.max(avg)]))
            if extremum != 0:
                avg = avg / extremum

        return avg

    def avg_ta(self, size=-1, maskOn=True, norm=None, cutoff=10):
        """
        Compute average TA curves of substacks of given size.

        Parameters
        ----------
        size : int, optional
            Size of substacks. If -1 average over whole stack is computed.
            The default is -1.
        maskOn : Boolean, optional
            Use mask for computation. The default is True.
        norm : str, optional
            Use norm for avg computation, i.e. minmax. The default is None.
        cutoff : int, optional
            If substack has less than cutoff non-zero pixel this substack in
            particular is discarded. The default is 10.

        Returns
        -------
        ta_curves : list of np array of float
            List of averaged TA curves.
        """
        # average over whole stack
        if size == -1:
            ta_curves = [self.avg(norm=norm, maskOn=maskOn)]
        # average over substacks of given size
        else:
            # compute substacks and average TA curves
            substacks = self.substacks(size=size, cutoff=cutoff)
            ta_curves = [i.avg(norm=norm, maskOn=maskOn) for i in substacks]

            return ta_curves

    def avg_show(self, maskOn=True, norm=None):
        fig, ax = plt.subplots()
        ax.plot(self.times, self.avg(maskOn=maskOn, norm=norm))
        plt.show()

    def project(self, maskOn=True):
        """
        Compute projection (sum of abs) of pp stack.

        Parameters
        ----------
        maskOn : Boolean, optional
            Use mask. The default is True.

        Returns
        -------
        result : 2d nparray
            Projection (sum and abs) of pp stack.

        """
        result = np.zeros(self.image_dimensions, dtype=np.float64)
        for i in self.images:
            np.add(result, np.abs(i), out=result)

        if maskOn is True:
            result = result * self.mask

        return result

    def total(self, maskOn=True):
        """
        Compute sum (no abs values) of stack.

        Parameters
        ----------
        maskOn : Boolean, optional
            Use mask. The default is True.

        Returns
        -------
        total : 2d np array
            Sum of all images in stack.

        """
        # compute sum of pp stack
        total = np.sum(self.images, axis=0)

        # apply mask
        if maskOn is True:
            total = np.where(self.mask, total, 0)

        return total

    def project_show(self, maskOn=True, export=None):
        """
        Display pp stack projection.

        Parameters
        ----------
        maskOn : Boolean, optional
            Use mask. The default is True.
        export : str, optional
            Saves plot into export. The default is None.

        Returns
        -------
        None.

        """
        # Choose a base colormap
        base_cmap = mlp.cm.get_cmap("viridis")

        # Create a new colormap from the base, setting the first color (for 0)
        # to white
        new_colors = base_cmap(np.linspace(0, 1, 256))
        new_colors[0, :] = np.array([1, 1, 1, 1])  # RGBA for white
        new_cmap = mlp.colors.ListedColormap(new_colors)

        fig, ax = plt.subplots()
        cax = ax.imshow(self.project(maskOn=maskOn), cmap=new_cmap)
        fig.colorbar(cax)

        if export is None:
            plt.show()
        else:
            plt.savefig(export, transparent=True)
            plt.show()

    def mask_slices(self, slices):
        """
        Set mask of slices to 0.

        Parameters
        ----------
        slices : list of slices
            Areas to mask out in, i.e. array(
                [[slice(None, None, None), slice(None, None, None)]],
                dtype=object
                )

        Returns
        -------
        None.

        """
        if slices is not None:
            for i in slices:
                self.mask[i] = False

    def count_nonzero_pixel(self):
        return np.count_nonzero(self.project())

    def substacks(self, size, cutoff=0):
        """
        Retrun substacks of size.

        Parameters
        ----------
        size : int
            Length of substack.
        cutoff : int, optional
            If substack has less than cutoff non-zero pixel this substack in
            particular is discarded. The default is 0.

        Returns
        -------
        stacks : TYPE
            DESCRIPTION.

        """
        stacks = []
        [index_x, index_y] = self._substack_index(size)

        for i in index_x:
            for j in index_y:
                temp_stack = PPS(
                    [self.images[:, i[0] : i[1], j[0] : j[1]], self.times],
                    dataType="data",
                    filename=self.filename,
                    mask=self.mask[i[0] : i[1], j[0] : j[1]],
                )
                if temp_stack.count_nonzero_pixel() >= cutoff:
                    stacks.append(temp_stack)

        return stacks

    def downsample(self, size):
        """
        Downsample stack by factor size.

        Parameters
        ----------
        size : int
            Size of new pixel in downsampled stack.

        Returns
        -------
        PPS stack
            Downsampled stack with reduced resolution and better SNR.

        """
        # create indicies for avergaing and an empty template
        erg = []
        mask = []
        [index_x, index_y] = self._substack_index(size)
        len_x = len(index_x)
        len_y = len(index_y)

        # loop over k(time delays), x(i) and y(j) and average
        for k in range(0, len(self.times)):
            temp = np.zeros((len_x, len_y), dtype=float)
            for i in range(0, len_x):
                for j in range(0, len_y):
                    block = self.images[
                        k,
                        index_x[i][0] : index_x[i][1],
                        index_y[j][0] : index_y[j][1],
                    ]
                    temp[i, j] = np.mean(block)
            erg.append(temp)

        mask = np.full((len_x, len_y), fill_value=True)
        for i in range(0, len_x):
            for j in range(0, len_y):
                block = self.mask[
                    index_x[i][0] : index_x[i][1],
                    index_y[j][0] : index_y[j][1],
                ]
                mask[i, j] = np.any(block)

        print(mask)
        try:
            print(mask.dtype)
        except:
            print("mask is apparently not an np array")
        print(type(mask))

        return PPS(
            [erg, self.times],
            dataType="data",
            filename=self.filename,
            mask=mask,
        )

    def classify(self, classifier, downsample=1, norm="minmax"):
        # generate 1 pixel stacks of downasmpled stack
        stack_ds = self.downsample(downsample)
        stacks = stack_ds.substacks(1, cutoff=-1)

        # writes all classes into instance attribute
        self.results["pigments"] = classifier.classes_

        # initialize counter, traces and color coding for each class
        traces = []
        counter_classes = {}
        color_numerical = {}
        j = 1
        for i in classifier.classes_:
            counter_classes[i] = 0
            color_numerical[i] = j
            j += 1

        # classify and count classifications
        for i in stacks:
            temp = i.avg(norm=norm)
            if np.abs(np.sum(temp)) == 0:
                traces.append(0)
            else:
                status = classifier.predict([temp])[0]
                for j in classifier.classes_:
                    if status == j:
                        counter_classes[j] += 1
                        traces.append(color_numerical[j])
                        break

        # re-shape array back into matrix form
        result_matrix = np.array(traces).reshape(stack_ds.image_dimensions)

        # compute ratios in pixel stats
        total = 0
        for i in classifier.classes_:
            total += counter_classes[i]
        for i in classifier.classes_:
            counter_classes[i + "_n"] = counter_classes[i] / total
        self.results["stats"] = counter_classes
        self.results["matrix"] = result_matrix

    def classify_accuracy(self, correct_classes):
        # check if stack has been classified already
        correct_percentage = 0
        correct_identified = []
        pigments = self.results["pigments"]
        sorted_subset = sorted(
            pigments, key=lambda k: self.results["stats"][k], reverse=True
        )
        counter = 0
        if self.results == {}:
            print("stack has not been classified")
        else:
            for i in correct_classes:
                correct_percentage += self.results["stats"][i + "_n"]
                if i in sorted_subset[:2]:
                    counter += 1
                    correct_identified.append(i)

        self.results["stats"]["correct_identified"] = correct_identified
        self.results["stats"]["correct_percentage"] = correct_percentage

    def classify_show(
        self, classifier, downsample=1, norm="minmax", export=None, alpha=1
    ):
        n = len(classifier.classes_)

        # classify stack with classifier
        self.classify(classifier, downsample=downsample, norm=norm)

        # generate false coloring
        set1_colors = [
            "white",
            "#d62728",
            "#1f77b4",
            "#ff7f0e",
            "#8c564b",
            "#2ca02c",
            "#17becf",
            "gold",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
        ]
        cmap = mlp.colors.ListedColormap(set1_colors[: n + 1])

        # create label
        label = []
        for i in classifier.classes_:
            a = self.results["stats"][i + "_n"]
            label.append(i + " " + f"{a*100:.2f} %")
        label = np.insert(label, 0, "nothing")

        # plotting
        fig, ax = plt.subplots()
        im = ax.matshow(
            self.results["matrix"],
            cmap=cmap,
            vmin=-0.5,
            vmax=n + 0.5,
            alpha=alpha,
        )
        cax = fig.colorbar(im, ticks=np.arange(0, n + 1))
        cax.set_ticks(np.arange(0, n + 1), labels=label)
        # im = ax.imshow([[0,1,2],[3,3,5],[6,7,0]], cmap=cmap)
        # cbar = fig.colorbar(im, ax=ax)
        # cbar.set_ticks(np.linspace(0.5, n-0.5, num=8), labels=label)
        # cbar = plt.colorbar(cax, ticks=np.linspace(1, n+2, num=n+1),
        #                    fraction=0.046, pad=0.1)
        # cbar.ax.set_yticklabels(label)
        ax.set_title(self.filename)

        if export is None:
            plt.show()
        else:
            plt.savefig(export, transparent=True)
            plt.show()

    def phasor(self, freq=0.25, remove_zero=False):
        """
        Compute phasor.

        Parameters
        ----------
        freq : TYPE, optional
            phasor frequency in THz. The default is 0.25.
        remove_zero : TYPE, optional
            remove all zero-pixel. The default is False.

        Returns
        -------
        TYPE
            list of phasor coordinates.

        """
        # define sin, cos of time delays, and prepare list containing TA curve
        # of each pixel
        self.freq = freq * 2 * np.pi
        self.sin = np.sin(self.times * self.freq)
        self.cos = np.cos(self.times * self.freq)
        self.ta_curves = self._phasor_flatten_stack(remove_zero=remove_zero)

        self.phasor_coor = np.apply_along_axis(
            self._phasor_compute, axis=1, arr=self.ta_curves
        )

        return self.phasor_coor

    def phasor_show(self, freq=0.25, remove_zero=True, color="red"):
        """
        Compute and show phasor.

        Parameters
        ----------
        freq : TYPE, optional
            phasor frequency in THz. The default is 0.25.
        remove_zero : TYPE, optional
            remove all zero-pixel. The default is True.
        color : TYPE, optional
            color of phasor plot. The default is "red".

        Returns
        -------
        None.

        """
        # compute phasor
        self.phasor(freq=freq, remove_zero=remove_zero)

        # compute semi circle
        theta = np.linspace(-(np.pi) / 2, np.pi / 2, 100)
        x1 = (1 - np.sin(theta)) / 2
        y1 = np.cos(theta) / 2
        x2 = (-1 + np.sin(-theta)) / 2
        y2 = -np.cos(-theta) / 2
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        # define false color scheme
        colors = [
            (plt.cm.colors.to_rgba(color, alpha)) for alpha in np.linspace(0, 1, 256)
        ]
        cmapp = mlp.colors.LinearSegmentedColormap.from_list("transparent_red", colors)

        fig, ax = plt.subplots(1, 1)

        # plot semicircle
        ax.plot(x, y, linestyle="dashed", color="grey")

        # plot phasor histogram
        ax.hist2d(
            self.phasor_coor[:, 0],
            self.phasor_coor[:, 1],
            bins=(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01)),
            cmap=cmapp,
        )

        # other plot settings
        ax.set_aspect("equal")  # Set the aspect ratio to equal
        ax.grid(True)
        ax.set_xlabel("g")
        ax.set_ylabel("s")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title("phasor frequency: " + str(self.freq / 2 / np.pi))
        plt.show()

    def _phasor_flatten_stack(self, remove_zero=False):
        # flatten images into list of TA curves
        ta_curves = self.images.reshape(len(self.times), -1).T
        if remove_zero is True:
            ta_curves = ta_curves[np.all(ta_curves != 0, axis=1)]

        return ta_curves

    def _phasor_compute(self, ta_curve):
        norm = np.sum(np.abs(ta_curve))
        phasor_coor = np.array(
            [
                np.dot(self.cos, ta_curve) / norm,
                np.dot(self.sin, ta_curve) / norm,
            ]
        )
        return phasor_coor

    def intensity_threshold(
        self, threshold="Li", sigma=5, projection_use_mask=True, show_mask=False
    ):
        """
        Compute intensity threshold mask.

        Compute stack projection with absolute value and derive intensity
        thresholded mask, either by Li threshold or by numeric value threshold.

        Parameters
        ----------
        threshold : str or number, optional
            If Li a Li threshold cutoff is used, if numeric this nymber is used
            as cutoff for the mask. The default is 'Li'.
        sigma : numeric, optional
            Gaussian filter sigma. The default is 5.
        projection_use_mask : boolean, optional
            If False the mask of this stack is not used for the projection.
            The default is True.
        show_mask : boolean, optional
            Plot mask if True. The default is False.

        Returns
        -------
        None.

        """
        # import filter
        from skimage import filters

        # compute intensity projection and do gaussian smoothing
        projection = filters.gaussian(
            self.project(maskOn=projection_use_mask), sigma=sigma
        )

        # compute mask
        if threshold == "Li":
            cutoff = filters.threshold_li(projection)
            self.mask = self.mask & np.where(projection > cutoff, True, False)
        elif isinstance(threshold, (int, float)):
            self.mask = self.mask & np.where(projection > threshold, True, False)
        else:
            print("invalid use of threshold variable")

        if show_mask:
            self.mask_show()

    @staticmethod
    def intensity_threshold_shared(stacks, threshold, sigma=5):
        """
        Return intensityold mask based on multiple pump-probe stacks.

        Absolute value of all images in all stacks are summed up, Gaussian
        filtered and intensity thresholded with cutoff.

        Parameters
        ----------
        stacks : list of stacks
            list of stacks.
        threshold : number
            cutoff threshold.
        sigma : float, optional
            Sigma value for Gaussian filter. The default is 5.

        Returns
        -------
        array of booleans
            Intensity mask.

        """
        from skimage.filters import gaussian

        # combine all images into single array
        all_images = np.concatenate([i.images for i in stacks])

        # compute projection
        projection = gaussian(
            np.sum([np.abs(i) for i in all_images], axis=0), sigma=sigma
        )

        # return mask
        return np.where(projection > threshold, True, False)

    def mask_show(self):
        """
        Show mask and masked projection of stack.

        Returns
        -------
        None.

        """
        # --- Custom colormap for the "normal" image ---
        base_cmap = mlp.cm.get_cmap("viridis")
        new_colors = base_cmap(np.linspace(0, 1, 256))
        new_colors[0, :] = np.array([1, 1, 1, 1])  # lowest value → white
        new_cmap = mlp.colors.ListedColormap(new_colors)

        # --- Discrete colormap for mask ---
        low_color = mlp.cm.get_cmap("viridis")(0.0)  # violet/blue end
        high_color = new_cmap(1.0)  # yellow end
        cmap_mask = mlp.colors.ListedColormap([low_color, high_color])

        bounds = [0, 1, 2]
        norm_mask = mlp.colors.BoundaryNorm(bounds, cmap_mask.N)

        # --- Side-by-side plots ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: binary mask
        cax1 = axes[0].imshow(self.mask, cmap=cmap_mask, norm=norm_mask)
        axes[0].set_title("Mask")
        axes[0].axis("off")
        fig.colorbar(cax1, ax=axes[0], ticks=[0, 1], fraction=0.046, pad=0.04)

        # Right: normal image with custom cmap
        cax2 = axes[1].imshow(self.project(), cmap=new_cmap)
        axes[1].set_title("Image")
        axes[1].axis("off")
        cb2 = fig.colorbar(cax2, ax=axes[1], fraction=0.046, pad=0.04)

        # Put ticks at the edges of the colorbar
        vmin, vmax = cax2.get_clim()
        cb2.set_ticks([vmin, vmax])
        cb2.ax.set_yticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

        plt.tight_layout()
        plt.show()

    def mask_update(self, mask):
        """
        Update mask of stack with mask.

        The mask propperty of this stack is updated (logical and) with mask.

        Parameters
        ----------
        mask : arry of bool
            Mask to be updtaed with.

        Returns
        -------
        None.

        """
        self.mask = self.mask & mask

    @staticmethod
    def linear_combination(stack1, coeff1, stack2, coeff2):
        """
        Compute linear combination of stack1 and stack2.

        Parameters
        ----------
        stack1 : pp stack
            stack1 for linear combination.
        coeff1 : numeric
            coefficient for stack1.
        stack2 : pp stack
            stack2 for linear combination.
        coeff2 : TYPE
            coefficient for stack2.

        Returns
        -------
        pp stack
            linear combination coeff1 * stack1 + coeff2 * stack2.

        """
        if np.allclose(stack1.times, stack2.times):
            images = coeff1 * stack1.images + coeff2 * stack2.images
            mask = stack1.mask & stack2.mask
            return PPS([images, stack1.times], mask=mask)
        else:
            print("time delays of stack1 and stack2 differ")
            return None


def main():
    # path1 = ("W:/Data/CutaneousMelanoma/sample_set_Elpis/N93/"
    #          + "20210628_770_730_N93_1_C01-ROI03_evaluation/")

    path1 = "W:\\Data\\CutaneousMelanoma\\sample_set_Elpis\\P75\\20210728_770_730_P75_1_C01-ROI07"
    filename = path1 + "preProcessed_v032.erg"

    # os.chdir("C:/Users/david/OneDrive/todo/programming/python/python")
    os.chdir(path1)

    t1 = PPS("preProcessed_v032.erg")
    print(t1.count_nonzero_pixel())

    phasor1 = t1.phasor(freq=0.25, remove_zero=True)
    phasor2 = t1.phasor(freq=0.1, remove_zero=True)
    fig, ax = plt.subplots(1, 1)
    ax.plot(phasor1[:, 0], phasor1[:, 1])
    ax.plot(phasor2[:, 0], phasor2[:, 1])

    plt.show()
    print(t1.phasor(remove_zero=True))
    # p1 = t1.phasor(remove_zero=True)
    # print(len(p1), p1)

    # fig, ax = plt.subplots(1, 1)
    # for i in t1.ta_curves[:10]:
    #   ax.plot(t1.times, i)
    # plt.show()


if __name__ == "__main__":
    main()
