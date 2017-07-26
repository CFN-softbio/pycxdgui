import numpy as np

# Calibration
################################################################################
class Calibration:
    """
    The geometric claculations used here are described in Yang, J Synch Rad (2013) 20, 211–218
    http://dx.doi.org/10.1107/S0909049512048984
    (See Appendix A)

    For the rotations, let's assume that the detector is initially
    oriented in the x-y plane, with the xray beam traveling along the
    negative z direction (as figure 5 in paper).

    The symbols in the paper translate to the following variables:
        chi : det_phi
            This can be thought of the orientation of the detector about
            the beam center (with all other angles zero).
            A positive motion of this results in a counter-clockwize
            rotation of the detector in the x-y plane (or a clockwise
            rotation of the coordinates).
        phi : det_orient
            this angle specifies angle of the axis with which the
            detector tilts. The tilt itself is specified by tau (next
            variable).
        tau : det_tilt
            This is the amount of tilt of the detector about some axis
            that is in the plane of the detector. The axis has an extra
            degree of freedom defined by p
            defined. It appears to be the negative of the axis defined
            in the paper (TODO : to verify)
            Note that no tilt (in tau) results in no movement
                (regardless of phi).


    Just to rephrase what is said in paper. When he says that chi (or
    det_phi) is only for "grazing-incident measurements", he means that
    the data will not be azimuthally averaged. Note that there are still
    plenty of SAXS measurements where this tilt is still important.

    """
    def __init__(self, wavelength_A=None, distance_m=None, pixel_size_um=None,
                 x0=None, y0=None, width=None, height=None, det_orient=0.,
                 det_tilt=0., det_phi=0., incident_angle=0., sample_normal=0.):

        self.det_orient = det_orient
        self.det_tilt = det_tilt
        self.det_phi = det_phi

        self.incident_angle = incident_angle
        self.sample_normal = sample_normal


        self.rot_matrix = None

        self.wavelength_A = wavelength_A
        self.distance_m = distance_m
        self.pixel_size_um = pixel_size_um
        # TODO :should we force these to be defined?
        # or add another object inheritance layer?
        if width is None:
            self.width = None
        else:
            self.width = width

        if height is None:
            self.height = None
        else:
            self.height = height

        if x0 is None:
            self.x0 = None
        else:
            self.x0 = x0

        if y0 is None:
            self.y0 = None
        else:
            self.y0 = y0


        # Data structures will be generated as needed
        # (and preserved to speedup repeated calculations)
        self.clear_maps()


    # Experimental parameters
    ########################################

    def set_angles(self, det_orient=0, det_tilt=0, det_phi=0, incident_angle=0., sample_normal=0.):
        self.det_orient = det_orient
        self.det_tilt = det_tilt
        self.det_phi = det_phi

        self.incident_angle = incident_angle
        self.sample_normal = sample_normal

    def get_ratioDw(self):
        ''' ratio of sample to detector distance to width.'''
        width_mm = self.width*self.pixel_size_um/1000.
        return self.distance_m/(width_mm/1000.)

    # Maps
    ########################################

    @property
    def q_map(self):
        '''Returns a 2D map of the q-value associated with each pixel position
        in the detector image.'''

        if self.q_map_data is None:
            self.generate_maps()

        return self.q_map_data

    @property
    def angle_map(self):
        '''Returns a map of the angle for each pixel (w.r.t. origin).
        0 degrees is vertical, +90 degrees is right, -90 degrees is left.'''

        if self.angle_map_data is not None:
            self.generate_maps()

        return self.angle_map_data

    @property
    def qx_map(self):
        if self.qx_map_data is None:
            self.generate_maps()

        return self.qx_map_data

    @property
    def qy_map(self):
        if self.qy_map_data is None:
            self.generate_maps()

        return self.qy_map_data

    @property
    def qz_map(self):
        if self.qz_map_data is None:
            self.generate_maps()

        return self.qz_map_data

    @property
    def qr_map(self):
        if self.qr_map_data is None:
            self.generate_maps()

        return self.qr_map_data

    # r_map already defined in parent object


    def generate_maps(self):
        """
        calculate all coordinates (pixel position as well as various derived values)
        all coordinates are stored in 2D arrays, as is the data itself in Data2D
        """
        print("Generating qmaps (expensive computation)")
        self.calc_rot_matrix()

        (w,h) = (self.width, self.height)
        # y is columnds, x is rows
        self.Y, self.X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        X = self.X.ravel()
        Y = self.Y.ravel()

        (Q, Phi, Qx, Qy, Qz, Qr, Qn, FPol, FSA) = self.calc_from_XY(X, Y, calc_cor_factors=True)

        self.q_map_data = Q.reshape((h, w))

        self.qx_map_data = Qx.reshape((h, w))
        self.qy_map_data = Qy.reshape((h, w))
        self.qz_map_data = Qz.reshape((h, w))

        self.qr_map_data = Qr.reshape((h, w))
        self.qz_map_data = Qn.reshape((h, w))
        self.angle_map_data = np.degrees( Phi.reshape((h, w)) )
        self.FPol_map_data = FPol.reshape((h, w))
        self.FSA_map_data = FSA.reshape((h, w))




    # q calculation
    ########################################

    def calc_from_XY(self, X, Y, calc_cor_factors=False):
        """
        calculate Q values from pixel positions X and Y
        X and Y are 1D arrays
        returns reciprocal/angular coordinates, optionally returns
        always calculates Qr and Qn, therefore incident_angle needs to be set
        Note that Phi is saved in radians; but the angles in ExpPara are in degrees
        """
        if self.rot_matrix is None:
            raise ValueError('the rotation matrix is not yet set.')

        # the position vectors for each pixel, origin at the postion of beam impact
        # R.shape should be (3, w*h), but R.T is more convinient for matrix calculation
        # RT.T[i] is a vector
        # x0, y0 is considerence the origin, which is also beam center
        RT = np.vstack((X - self.x0, -(Y - self.y0), 0.*X))

        # get distance from sample in number of pixels (detector coordinates)
        dr = self.get_ratioDw()*self.width
        # position vectors in lab coordinates, sample at the origin
        [X1, Y1, Z1] = np.dot(self.rot_matrix, RT)
        Z1 -= dr

        # angles
        r3sq = X1*X1+Y1*Y1+Z1*Z1
        r3 = np.sqrt(r3sq)
        r2 = np.sqrt(X1*X1+Y1*Y1)
        Theta = 0.5*np.arcsin(r2/r3)
        Phi = np.arctan2(Y1, X1) + np.radians(self.sample_normal)

        Q = 2.0*self.get_k()*np.sin(Theta)

        # lab coordinates
        Qz = Q*np.sin(Theta)
        Qy = Q*np.cos(Theta)*np.sin(Phi)
        Qx = Q*np.cos(Theta)*np.cos(Phi)

        # convert to sample coordinates
        alpha = np.radians(self.incident_angle)
        Qn = Qy*np.cos(alpha) + Qz*np.sin(alpha)
        Qr = np.sqrt(Q*Q-Qn*Qn)*np.sign(Qx)

        if calc_cor_factors==True:
            FPol = (Y1*Y1+Z1*Z1)/r3sq
            FSA = np.power(np.fabs(Z1)/r3, 3)
            return (Q, Phi, Qx, Qy, Qz, Qr, Qn, FPol, FSA)
        else:
            return (Q, Phi, Qx, Qy, Qz, Qr, Qn)


    # Rotation calculation
    ########################################


    def RotationMatrix(self, axis, angle):
        if axis=='x' or axis=='X':
            rot = np.asarray(
                [[1., 0., 0.],
                [0., np.cos(angle), -np.sin(angle)],
                [0., np.sin(angle),  np.cos(angle)]])
        elif axis=='y' or axis=='Y':
            rot = np.asarray(
                [[ np.cos(angle), 0., np.sin(angle)],
                [0., 1., 0.],
                [-np.sin(angle), 0., np.cos(angle)]])
        elif axis=='z' or axis=='Z':
            rot = np.asarray(
                [[np.cos(angle), -np.sin(angle), 0.],
                [np.sin(angle),  np.cos(angle), 0.],
                [0., 0., 1.]])
        else:
            raise ValueError('unknown axis %s' % axis)

        return rot


    def calc_rot_matrix(self):

        # First rotate detector about x-ray beam
        tm1 = self.RotationMatrix('z', np.radians(-self.det_orient))

        # Tilt detector face (so that it is not orthogonal to beam)
        tm2 = self.RotationMatrix('y', np.radians(self.det_tilt))

        # Rotate detector about x-ray beam
        tm3 = self.RotationMatrix('z', np.radians(self.det_orient+self.det_phi))
        self.rot_matrix = np.dot(np.dot(tm3, tm2), tm1)

    # Experimental parameters
    ########################################
    def set_wavelength(self, wavelength_A):
        '''Set the experimental x-ray wavelength (in Angstroms).'''

        self.wavelength_A = wavelength_A


    def get_wavelength(self):
        '''Get the x-ray beam wavelength (in Angstroms) for this setup.'''

        return self.wavelength_A


    def set_energy(self, energy_keV):
        '''Set the experimental x-ray beam energy (in keV).'''

        energy_eV = energy_keV*1000.0
        energy_J = energy_eV/6.24150974e18

        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s

        wavelength_m = (h*c)/energy_J
        self.wavelength_A = wavelength_m*1e+10


    def get_energy(self):
        '''Get the x-ray beam energy (in keV) for this setup.'''

        h = 6.626068e-34 # m^2 kg / s
        c = 299792458 # m/s

        wavelength_m = self.wavelength_A*1e-10 # m
        E = h*c/wavelength_m # Joules

        E *= 6.24150974e18 # electron volts

        E /= 1000.0 # keV

        return E


    def get_k(self):
        '''Get k = 2*pi/lambda for this setup, in units of inverse Angstroms.'''

        return 2.0*np.pi/self.wavelength_A


    def set_distance(self, distance_m):
        '''Sets the experimental detector distance (in meters).'''

        self.distance_m = distance_m


    def set_pixel_size(self, pixel_size_um=None, width_mm=None, num_pixels=None):
        '''Sets the pixel size (in microns) for the detector. Pixels are assumed
        to be square.'''

        if pixel_size_um is not None:
            self.pixel_size_um = pixel_size_um

        else:
            if num_pixels is None:
                num_pixels = self.width
            pixel_size_mm = width_mm*1./num_pixels
            self.pixel_size_um = pixel_size_mm*1000.0


    def set_beam_position(self, x0, y0):
        '''Sets the direct beam position in the detector images (in pixel
        coordinates).'''

        self.x0 = x0
        self.y0 = y0

    @property
    def origin(self):
        return self.x0, self.y0


    def set_image_size(self, width, height=None):
        '''Sets the size of the detector image, in pixels.'''

        self.width = width
        if height is None:
            # Assume a square detector
            self.height = width
        else:
            self.height = height


    @property
    def q_per_pixel(self):
        '''Gets the delta-q associated with a single pixel. This is computed in
        the small-angle limit, so it should only be considered approximate.
        For instance, wide-angle detectors will have different delta-q across
        the detector face.'''

        if not hasattr(self, '_q_per_pixel') or self._q_per_pixel is not None:
            return self._q_per_pixel

        c = (self.pixel_size_um/1e6)/self.distance_m
        twotheta = np.arctan(c) # radians

        self._q_per_pixel = 2.0*self.get_k()*np.sin(twotheta/2.0)

        return self._q_per_pixel


    # Maps
    ########################################

    def clear_maps(self):
        self.r_map_data = None
        self._q_per_pixel = None
        self.q_map_data = None
        self.angle_map_data = None

        self.qx_map_data = None
        self.qy_map_data = None
        self.qz_map_data = None
        self.qr_map_data = None



    @property
    def r_map(self):
        '''Returns a 2D map of the distance from the origin (in pixel units) for
        each pixel position in the detector image.'''

        if self.r_map_data is not None:
            return self.r_map_data

        x = np.arange(self.width) - self.x0
        y = np.arange(self.height) - self.y0
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        self.r_map_data = R

        return self.r_map_data

