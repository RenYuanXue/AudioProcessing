"""
Provide functions used in features implementation
Provides: xxxx

Author: Ren Yuan (Peter) Xue
"""

import numpy as np
import scipy.fftpack as fft
from scipy import signal
import spectrum

def get_num_fft(sample_rate, window_len):
    """
    Function get_num_fft calculates optimal number of FFT points based on frame length.
    Less number of FFT points than length of frame
    will lose precision by droppping many of the samples.
    Therefore, we want num_fft as a power of 2, greater than frame length.
    
    @param sample_rate: The sample rate of audio signal we working with.
    @param window_len: Time interval we are taking within frames.
    @returns: Optimal number of FFT points.
    """
    frame_length = sample_rate * window_len
    num_fft = 1
    while num_fft < frame_length:
        num_fft *= 2
    return num_fft


def powspec(signal, sample_rate, window_len, hop_size, num_fft):
    """
    Function powspec produces the power spectrum of the given audio signal 
    
    @param signal: Audio signal we are working with.
    @param sample_rate: The sample rate of our audio signal.
    @param window_len: Time interval we are taking within frames.
    @param hop_size: Time step we are taking between frames.
    @param num_fft: Number of FFT points.
    @returns: A Power spectrum.
    """
    # Convert from seconds to samples.
    frame_length, frame_stride = window_len * sample_rate, hop_size * sample_rate
    frame_length, frame_stride = int(round(frame_length)), int(round(frame_stride))
    signal_length = len(signal)
    # Make sure that we have at least 1 frame.
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_stride))
    pad_signal_length = num_frames * frame_stride + frame_length
    diff = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal.
    pad_signal = np.append(signal, diff)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Apply Hamming window to frames.
    frames *= np.hamming(int(round(sample_rate * window_len)))
    # Calculate the Power Spectrum of the frames.
    magnitude_frames = np.absolute(np.fft.rfft(frames, num_fft))
    power_frames = ((1.0 / num_fft) * (magnitude_frames) ** 2)
    energy = np.log(sum(power_frames)) # Calculate log energy.
    return power_frames, energy


def get_filter(freq_min, freq_max, num_mel_filter, num_fft, sample_rate):
    """
    Function get_filter_points calculates where the filters in filter bank locate
    
    @param freq_min: Lowest frequency band edge of Mel filters in Hz.
    @param freq_max: Highest frequency band edge of Mel filters in Hz.
    @param num_mel_filter: Number of filter points in filter banks on Mel scale.
    @param num_fft: Number of FFT points.
    @param sample_rate: The sample rate of audio signal we working with.
    @returns: Filters used for computing filter bank feature.
    """
    # Get filter points.
    freq_min_mel = hz2mel(freq_min)
    freq_max_mel = hz2mel(freq_max)
    mels = np.linspace(freq_min_mel, freq_max_mel, num=num_mel_filter+2)
    freqs = mel2hz(mels)
    filter_points = np.floor((num_fft + 1) / sample_rate * freqs).astype(int)
    # Get filter bank filters.
    filters = np.zeros((len(filter_points)-2, int(num_fft/2+1)))
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n+1]] = np.linspace(0, 1, filter_points[n+1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n+2] - filter_points[n+1])
    return filters


def pre_emphasis(signal, coef):
    """
    Function pre-emphasis applies pre-emphasis filter 
    on the signal to amplify the high frequencies.
    
    @param signal: Audio signal.
    @param coef: Coefficient used in pre-empahsis filter.
    @returns: Pre-emphasized signal after applying the filter.
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def hz2mel(freq):
    """
    Function hz2mel calculates Mel values.
    
    @param freq: Frequency.
    @returns: Corresponding Mel value for given frequency.
    """
    return 2595.0 * np.log10(1.0 + freq / 700.0)
    
    
def mel2hz(mels):
    """
    Function mel2hz calculates Hertz values.
    
    @param mel: Mel value.
    @returns: Corresponding Hertz value for given Mel value.
    """
    hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    return hz


def hz2bark(freq):
    """
    Function hz2bark calculates Bark scale. 
    Use Traunmueller-formula for f  >  200 Hz
              Linear mapping for f <=  200 Hz
    
    @param freq: Frequency.
    @returns: Corresponding Bark scale for given frequency.
    """
    z_less_200 = freq / 102.9
    z_greater_200 = 26.81 * freq / (1960 + freq) - 0.53
    return (freq > 200) * z_greater_200 + (freq <= 200) * z_less_200


def bark2hz(z):
    """
    Function bark2hz Using Hynek's formula to calculate calculate corresponding Hertz.
    
    @param z: Bark scale.
    @returns: corresponding Hertz to z.
    """
    hz = np.multiply(600, np.sinh(np.divide(z, 6)))
    return hz


def audspec(powspec,sample_rate=None,num_filters=None,fbtype='bark',freq_min=0,freq_max=None,sum_power=True,bandwidth=1.0):
    """
    Function audspec performs critical band analysis.
    
    @param powerspec: Power Spectrum.
    @param sample_rate: The sample rate of our audio signal.
    @param num_filters: Number of filters.
    @param fbtype: The frequency type we are working with.
    @param freq_min: Lowest frequency in Bark scale.
    @param freq_max: Highest frequency in Bark scale.
    @param sum_power: Integrate FFT bins into Mel bins, in sum_power domains:
    @param bandwidth: The critical bandwidth.
    @returns: Corresponding Mel value for given frequency.
    """
    # Handle exceptions.
    if not sample_rate: # Check sample_rate input validness.
        return('Invalid input for sample_rate')
    # Initialize parameters.
    freq_max = freq_max or int(sample_rate/2)
    num_filters = num_filters or np.ceil(hz2bark(sample_rate/2)) + 1
    num_freqs = powspec.shape[0]
    num_fft = (int(num_freqs) - 1) * 2
    # Consider frequency domains.
    if fbtype == 'bark':
        weight_matrix = fft2barkmx(num_fft, sample_rate, num_filters, bandwidth, freq_min, freq_max)
    elif fbtype == 'mel':
        weight_matrix = fft2melmx(num_fft, sample_rate, num_filters, bandwidth, freq_min, freq_max)
    elif fbtype == 'htkmel':
        weight_matrix = fft2melmx(num_fft, sample_rate, num_filters, bandwidth, freq_min, freq_max, 1, 1)
    elif fbtype == 'fcmel':
        weight_matrix = fft2melmx(num_fft, sample_rate, num_filters, bandwidth, freq_min, freq_max, 1, 0)
    else:
        return 'Invalid fbtype input'

    weight_matrix = weight_matrix[:, 0:num_freqs]
    # Integrate FFT bins into Mel bins, in abs (if sum_power = True) or abs^2 domains.
    if sum_power:
        aspectrum = np.matmul(weight_matrix, powspec)
    else:
        aspectrum = np.power((np.matmul(weight_matrix,np.sqrt(powspec))), 2)
    return aspectrum


def fft2barkmx(fft_length, fs, nfilts = 0, band_width = 1, min_freq = 0, max_freq = 0):
    """
    Function fft2barkmax generates a matrix of weights 
    to combine FFT bins into Bark bins.
    
    @param num_fft: Number of FFT points.
    @param sample_rate: The sample rate of our audio signal.
    @param num_filters: Number of filters. Default is 0.
    @param width: Constant width of each band in Bark. Default is 1.
    @param freq_min: Lowest frequency in Hertz. Default is 0.
    @param freq_max: Highest frequency in Hertz. Default is sample_rate / 2.
    @returns: A matrix of weights to combine FFT bins into Bark bins.
    """
    # Initialize parameters.
    if max_freq == 0:
        max_freq = fs / 2
        
    min_bark = hz2bark(min_freq)
    nyqbark = hz2bark(max_freq) - min_bark
    
    if nfilts == 0 :
        nfilts = np.ceil(nyqbark) + 1
    
    wts = np.zeros((int(nfilts), int(fft_length)))
    step_barks = nyqbark / (nfilts - 1)
    binbarks = hz2bark(np.arange(0, fft_length / 2 + 1) * fs / fft_length)
    
    for i in range (int(nfilts)):
        f_bark_mid = min_bark + np.multiply(i, step_barks)
        lof = np.subtract(np.subtract(binbarks, f_bark_mid), 0.5)
        hif = np.add(np.subtract(binbarks, f_bark_mid), 0.5)
        minimum = np.minimum(0, np.minimum(hif, np.multiply(-2.5, lof)) / band_width)
        wts[i, 0 : int(fft_length / 2) + 1] = np.power(10, minimum)
    return wts


def rasta_filter(x):
    """
    Function rasta_filter turns a (critical band by frame) matrix.
    Default filter is single pole at 0.94.
    
    @param x: Rows of x = critical bands, cols of x = frmes.
    @returns: A (critical band by frame) matrix.
    """
    # RASTA filter.
    numer = np.arange(-2, 3)
    numer = -numer / np.sum(numer ** 2)
    denom = np.array([1, -0.94])
    # Initialize the state. This avoids a big spike at the beginning
    # resulting from the dc oggrdt level in each band.
    zi = signal.lfilter_zi(numer,1)
    y = np.zeros((x.shape))
    # Dont keep any of these values, just output zero at the beginning.
    # Apply the full filter to the rest of the signal, append it.
    for i in range(x.shape[0]):
        y1, zi = signal.lfilter(numer, 1, x[i, 0:4], axis = 0, zi = zi * x[i, 0])
        y1 = y1*0
        y2, _ = signal.lfilter(numer, denom, x[i, 4:x.shape[1]], axis = 0, zi = zi)
        y[i, :] = np.append(y1, y2)
    return y


def postaud(x, freq_max, fbtype='bark', boarden=0):
    """
    Function postaud returns the compressed audio.
    Does loudness equalization and cube root compression.
    
    @param x: Critical band filters.
    @param freq_max: Highest frequency band edge in Hz.
    @param fbtype: The frequency domain we are working with. Default is 'bark'.
    @param boarden: Number of extra flanking bands. Default is 0.
    @returns: The cube root compressed audio.
    """
    num_bands, num_frames = x.shape
    num_fpts = int(num_bands + 2 * boarden) # Include frequency points at extremes, discard later.
    
    if fbtype == 'bark':
        bandcfhz = bark2hz(np.linspace(0, hz2bark(freq_max), num_fpts))
    elif fbtype == 'mel':
        bandcfhz = mel2hz(np.linspace(0, hz2mel(freq_max), num_fpts))
    else:
        return 'Invalid fbtype input'
    
    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[boarden : (num_fpts - boarden)]
    
    # Hynek's magic equal-loudness-curve formula
    fsq = np.power(bandcfhz, 2)
    ftmp = np.add(fsq, 1.6e5)
    eql = np.multiply(np.power(np.divide(fsq, ftmp), 2), np.divide(np.add(fsq, 1.44e6), np.add(fsq, 9.61e6)))
    # Weight the critical bands.
    z = np.multiply(np.tile(eql, (num_frames, 1)).T, x)
    # Cube root compress.
    z = np.power(z, 0.33)
    # Replicate first and last band (because they are unreliable as calculated).
    if boarden:
        y = np.zeros((z.shape[0] + 2, z.shape[1]))
        y[0, :] = z[0, :]
        y[1:num_bands + 1, :] = z
        y[num_bands + 1, :] = z[z.shape[0] - 1, :]
    else:
        y = np.zeros((z.shape[0], z.shape[1]))
        y[0, :] = z[1, :]
        y[1:num_bands - 1, :] = z[1:z.shape[0] - 1, :]
        y[num_bands - 1, :] = z[z.shape[0] - 2, :]
    
    return y, eql


def dolpc(x, model_order=8):
    """
    Function dolpc computes the autoregressive model from spectral magnitude samples.
    
    @param x: Critical band filters.
    @param model_order: Order of model. Default is 8.
    @returns: Autoregressive model from spectral magnitude samples.
    """
    num_bands, num_frames = x.shape
    
    # Calculate autocorrelation
    R = np.zeros((2 * (num_bands - 1), num_frames))
    R[0:num_bands, :] = x
    for i in range(num_bands - 1):
        R[i + num_bands - 1, :] = x[num_bands - (i + 1), :]
    r = fft.ifft(R.T).real.T
    r = r[0:num_bands, :]
    y = np.ones((num_frames, model_order + 1))
    e = np.zeros((num_frames, 1))
    
    # Find LPC coeffs by durbin
    if model_order == 0:
        for i in range(num_frames):
            _ , e_tmp, _ = spectrum.LEVINSON(r[:, i], model_order, allow_singularity = True)
            e[i, 0] = e_tmp
    else:
        for i in range(num_frames):
            y_tmp, e_tmp, _ = spectrum.LEVINSON(r[:, i], model_order, allow_singularity = True)
            y[i, 1:model_order + 1] = y_tmp
            e[i, 0] = e_tmp
    
    # Normalize each poly by gain.
    y = np.divide(y.T, np.add(np.tile(e.T, (model_order + 1, 1)), 1e-8))
    
    return y


def lpc2cep(a, nout = None):
    """
    Function lpc2cep converts the LPC 'a' coefficients in each column of lpcas
    into frames of cepstra.
    
    @param a: LPC.
    @param nout: Number of cepstra to produce. Defaults to len(a).
    """
    nin, ncol = a.shape
    order = nin - 1
    if not nout:
        nout = order + 1
    
    # First cep is log(Error) from Durbin.
    cep = np.zeros((nout, ncol))
    cep[0, :] = -np.log(a[0, :])
    # Renormalize LPC a coefficients.
    norm_a = np.divide(a, np.add(np.tile(a[0, :], (nin, 1)), 1e-8))
    
    for n in range(1, nout):
        total = 0
        for m in range(1, n):
            total = np.add(total, np.multiply(np.multiply((n - m), norm_a[m, :]), cep[(n - m), :]))
        
        cep[n, :] = -np.add(norm_a[n, :], np.divide(total, n))
    
    return cep


def lpc2spec(lpcas, nout=None):
    """
    Function lpc2spec converts LPC coefficients back into spectra.
    
    @param lpcas: LPC analysis.
    @param nout: Number of frequency channels. Dafault is 17 (i.e. for 8 kHz)
    @returns: The spectra coefficients.
    """
    nout = nout or 17

    rows, cols = lpcas.shape
    order = rows - 1
    gg = lpcas[1,:]
    aa = np.divide(lpcas, np.tile(gg, (rows,1)))

     # Calculate the actual z-plane polyvals: nout points around unit circle.
    tmp_1 = np.array(np.arange(0, nout), ndmin = 2).T
    tmp_1 = np.divide(np.multiply(-1j, np.multiply(tmp_1, np.pi)), (nout - 1))
    tmp_2 = np.array(np.arange(0, order + 1), ndmin = 2)
    zz = np.exp(np.matmul(tmp_1, tmp_2))

    # Actual polyvals, in power (mag^2).
    features = np.divide(np.power(np.divide(1, np.abs(np.matmul(zz, aa))), 2), np.tile(gg, (nout, 1)))

    F = np.zeros((cols, int(np.ceil(rows/2))))
    M = F

    for c in range(cols):
        aaa = aa[:, c]
        rr = np.roots(aaa)
        ff_tmp = np.angle(rr)
        ff = np.array(ff_tmp, ndmin = 2).T
        zz = np.exp(np.multiply(1j, np.matmul(ff, np.array(np.arange(0, aaa.shape[0]), ndmin = 2))))
        mags = np.sqrt(np.divide(np.power(np.divide(1, np.abs(np.matmul(zz, np.array(aaa, ndmin = 2).T))), 2), gg[c]))

        ix = np.argsort(ff_tmp)
        dummy = np.sort(ff_tmp)
        tmp_F_list = []
        tmp_M_list = []

        for i in range(ff.shape[0]):
            if dummy[i] > 0:
                tmp_F_list = np.append(tmp_F_list, dummy[i])
                tmp_M_list = np.append(tmp_M_list, mags[ix[i]])

        M[c, 0 : tmp_M_list.shape[0]] = tmp_M_list
        F[c, 0 : tmp_F_list.shape[0]] = tmp_F_list
        
    return features, F, M


def spec2cep(spec, ncep,dcttype):
    """
    Function spec2cep calculate cepstra from spectral samples (in columns of spec)
    
    @param spec: The input spectral samples.
    @param ncep: Number of cepstral. Default is 13.
    @param dcttype: Type of DCT.
    """
    nrow, ncol = spec.shape
    dctm = np.zeros((ncep, nrow))
    
    # Orthogonal one.
    if dcttype == 2 or dcttype == 3:
        for i in range(ncep):
            dctm[i, :] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(1, 2 * nrow, 2)), (2 * nrow)), np.pi)), np.sqrt(2 / nrow))
        # Make it unitary.
        if dcttype == 2:
            dctm[0, :] = np.divide(dctm[0, :], np.sqrt(2))
    #      
    elif dcttype == 4:
        for i in range(ncep):
            dctm[i, :] = np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(1, nrow + 1)), (nrow + 1)), np.pi)), 2)
            dctm[i, 0] = np.add(dctm[i, 0], 1)
            dctm[i, int(nrow - 1)] = np.multiply(dctm[i, int(nrow - 1)], np.power(-1, i))
        dctm = np.divide(dctm, 2 * (nrow + 1))
    # DPWE type 1 - expand and used fft.
    else:
        for i in range(ncep):
            dctm[i, :] = np.divide(np.multiply(np.cos(np.multiply(np.divide(np.multiply(i, np.arange(0, nrow)), (nrow - 1)), np.pi)), 2), 2 * (nrow - 1))
        dctm[:, 0] = np.divide(dctm[:, 0], 2)
        # Fixup 'non-repeated' points.
        dctm[:, int(nrow - 1)] = np.divide(dctm[:, int(nrow - 1)], 2)
    
    cep = np.matmul(dctm, np.log(np.add(spec, 1e-8)))
    
    return cep, dctm


def lifter(x, lift = 0.6, invs = False):
    """
    Function lifter applies lifter to matrix of cepstra (one per column)
    
    @param x: Matrix of cepstra.
    @param lift: Expont of x inverse liftering.
    @param invs: If inverse = True, undo the liftering. Default is False.
    """
    ncep = x.shape[0]
    
    if lift == 0:
        y = x
    else:
        if lift < 0:
            lift = 0.6
        liftwts = np.power(np.arange(1, ncep), lift)
        liftwts = np.append(1, liftwts)
        
        if (invs):
            liftwts = np.divide(1, liftwts)
        
        y = np.matmul(np.diag(liftwts), x)
    
    return y


