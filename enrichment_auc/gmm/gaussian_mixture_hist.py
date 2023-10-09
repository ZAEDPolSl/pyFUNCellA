import math
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


def sort(arr):
    # Given arr should be 1D
    arr_ind = np.arange(0, len(arr))
    dict_arr = dict(zip(arr_ind, arr))
    sorted_dict_arr = sorted(dict_arr.items(), key=lambda x: x[1])
    arr_sorted = np.array([pair[1] for pair in sorted_dict_arr])
    ind_after_sort = np.array([pair[0] for pair in sorted_dict_arr])
    return arr_sorted, ind_after_sort


def gaussian_mixture_hist(
    x,
    y,
    KS=10,
    cores=0,
    criterion="bic",
    path_visualisation_name=None,
    SW=0.25,
    stopping_threshold=0.2,
    min_clusters=5,
    n_clusters=None,
):
    TIC = sum(y)
    BIC = []
    logL = [i for i in range(0, KS)]
    D = []
    alpha = dict.fromkeys([i for i in range(0, KS)])
    mu = dict.fromkeys([i for i in range(0, KS)])
    sigma = dict.fromkeys([i for i in range(0, KS)])
    i_bias_factor = 0
    # handling vector mode in hist implementation
    if np.all((y == 1)):
        _x = x
        _y = y
        y, x = np.histogram(x, bins=max(min(20, round(math.sqrt(x.shape[0]))), 400))
        x = [(el[0] + el[1]) / 2 for el in list(zip(x, x[1:]))]
    else:
        _x = x
        _y = y

    # 1 component
    [alpha[0], mu[0], sigma[0], logL[0]] = EM_iter_hist(
        _x,
        _y,
        np.array([1]),
        np.array([np.mean(x)]),
        np.array([np.std(x, ddof=1)]),
        TIC,
        SW=SW,
    )
    bic = -2 * logL[0] + 2 * np.log(TIC)
    BIC.append(bic)

    # >2 components
    stop = False
    k = 2
    Nb = len(x)
    if cores == 0:
        aux_mx = dyn_pr_split_w_aux(x, y)
    else:
        _step = Nb // cores
        _missing = Nb % cores
        _start = [_step * i for i in range(0, cores)]
        _stop = [_step * i for i in range(1, cores + 1)]
        _stop[-1] = (
            _stop[-1] + _missing - 1
        )  # -1 because that was in original fun on the end

        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.starmap(
                dyn_pr_split_w_aux_core,
                [(x, y, _start[i], _stop[i]) for i in range(0, cores)],
            )
        aux_mx = sum(results)

    while not stop and k < KS:
        if n_clusters is not None:
            k = n_clusters
            cmp_nb = k - 1
            stop = True
        _, opt_part = dyn_pr_split_w(x, y, k - 1, aux_mx)

        part_cl = np.concatenate(([0], opt_part, [Nb + 1]), axis=0)  # 1 to 0
        pp_ini = np.zeros(
            k,
        )
        mu_ini = np.zeros(
            k,
        )
        sig_ini = np.zeros(
            k,
        )

        for kkps in range(0, k):
            invec = x[int(part_cl[kkps]) : int(part_cl[kkps + 1])]
            yinvec = y[int(part_cl[kkps]) : int(part_cl[kkps + 1])]

            wwec = yinvec / np.sum(yinvec)

            pp_ini[kkps] = np.sum(yinvec) / np.sum(y)  # yinvec and y are only 1D
            mu_ini[kkps] = np.sum(invec * wwec)
            sig_ini[kkps] = 0.5 * (np.max(invec) - np.min(invec))

        # decomposition
        alpha[k - 1], mu[k - 1], sigma[k - 1], logL[k - 1] = EM_iter_hist(
            _x, _y, pp_ini, mu_ini, sig_ini, TIC, SW=SW
        )

        # check convergence
        bic = -2 * logL[k - 1] + (3 * k) * np.log(TIC)
        BIC.append(bic)
        d = -2 * logL[k - 2] + 2 * logL[k - 1]
        D.append(d)

        # criterion choice
        if criterion == "bic":
            if 1 - stats.chi2.cdf(d, 3) > stopping_threshold and k > min_clusters:
                stop = True
        elif criterion == "bias_factor":
            bias_factor = np.exp((BIC[k - 2] - BIC[k - 1]) / 2)
            if bias_factor < 100:
                i_bias_factor = k - 1
                stop = True
        k += 1
    bic_arr = np.array(BIC)
    if n_clusters is None:
        if criterion == "bic":
            cmp_nb = np.unravel_index(bic_arr.argmin(), bic_arr.shape)[0]
        elif criterion == "bias_factor":
            cmp_nb = i_bias_factor

    pp_est = alpha[cmp_nb]
    mu_est = mu[cmp_nb]
    sig_est = sigma[cmp_nb]

    if path_visualisation_name is not None:
        plot_bic(BIC, cmp_nb, path_visualisation_name)

    return pp_est, mu_est, sig_est


def EM_iter_hist(x, y, alpha, mu, sig, TIC, SW=0.25):
    [x, ind] = sort(x)
    y = y[ind]
    N = len(y)
    sig2 = sig**2
    change = np.Inf
    count = 1
    KS = np.max(alpha.shape)
    # TODO: more emphasis on SW in documentation - add warning to look at the histogram before procceding
    SW = np.power((max(x) - min(x)) / (4 * KS), 2)
    min_sigdev = np.power((max(x) - min(x)) * SW / KS, 2)
    eps_change = 1e-4

    while change > eps_change and count < 10000:
        old_alpha = alpha.copy()
        old_sig2 = sig2.copy()
        sig = np.sqrt(sig2)

        f = np.zeros((KS, N))
        for a in range(0, KS):
            f[a, :] = normal_pdf(x, mu[a], sig[a])

        px = np.matmul(alpha, f)
        px[np.isnan(px)] = 5e-324
        px[px == 0] = 5e-324

        for a in range(0, KS):
            pk = (alpha[a] * f[a, :] * y) / px
            denom = np.sum(pk, axis=0)

            # mu
            mu[a] = np.matmul(pk, x) / (denom + 1e-15)
            sig2num = np.sum(np.matmul(pk, ((x - mu[a]) ** 2)))

            # sig
            if sig2num / denom < min_sigdev:
                sig2[a] = min_sigdev
            else:
                sig2[a] = sig2num / denom

            # alpha
            alpha[a] = denom / TIC

        change = np.sum(np.abs(alpha - old_alpha)) + np.sum(
            ((np.abs(sig2 - old_sig2)) / sig2)
        ) / (len(alpha))
        count += 1
    # returns
    logL = np.sum(np.log(px) * y)
    mu_est, ind = sort(mu)
    sig_est = np.sqrt(sig2[ind])
    pp_est = alpha[ind]

    return pp_est, mu_est, sig_est, logL


def dyn_pr_split_w_aux(data, ygreki):  # (x,y)
    N = len(data)
    aux_mx = np.zeros((N, N), dtype=np.float32)
    for kk in range(0, N - 1):
        for jj in range(kk + 1, N):
            aux_mx[kk, jj] = my_qu_ix_w(data[kk:jj], ygreki[kk:jj])
    return aux_mx


def dyn_pr_split_w_aux_core(data, ygreki, start, stop):
    N = len(data)
    aux_mx = np.zeros((N, N))
    for kk in range(start, stop):
        for jj in range(kk + 1, N):
            aux_mx[kk, jj] = my_qu_ix_w(data[kk:jj], ygreki[kk:jj])
    return aux_mx


def dyn_pr_split_w_aux_gpu(data, ygreki, aux_mx):  # (x,y)
    N = len(data)
    for kk in range(0, N - 1):
        for jj in range(kk + 1, N):
            invec = data[kk:jj]
            yinwec = ygreki[kk:jj]
            PAR = 1
            PAR_sig_min = 0.1
            if (invec[len(invec)] - invec[0]) <= PAR_sig_min or np.sum(
                yinwec
            ) <= 1.0e-3:
                wyn = np.inf
            else:
                wwec = yinwec / (np.sum(yinwec))
                wyn = (
                    PAR
                    + math.sqrt(np.sum(((invec - np.sum(invec * wwec)) ** 2) * wwec))
                ) / (
                    np.max(invec) - np.min(invec)
                )  # lots of matrix operations
            aux_mx[kk, jj] = wyn
    return aux_mx


def dyn_pr_split_w(data, ygreki, K_gr, aux_mx):
    # init
    Q = np.zeros((K_gr,))
    N = len(data)
    p_opt_idx = np.zeros((N,))
    p_aux = np.zeros((N,))
    opt_pals = np.zeros((K_gr, N))
    for kk in range(0, N):
        p_opt_idx[kk] = my_qu_ix_w(data[kk:N], ygreki[kk:N])
    # iter
    for kster in range(0, K_gr):
        for kk in range(0, N - kster - 1):
            for jj in range(kk + 1, N - kster):  # deleted ",N-kster+1"
                p_aux[jj] = aux_mx[kk, jj] + p_opt_idx[jj]
            holder_p_aux = p_aux[kk + 1 : N - kster]  # just for convenience purpose
            mm = np.min(holder_p_aux)
            ix = np.unravel_index(
                holder_p_aux.argmin(), holder_p_aux.shape
            )  # pos of min element
            # ix -> tuple (892,)
            p_opt_idx[kk] = mm
            opt_pals[kster, kk] = kk + ix[0] + 1
        Q[kster] = p_opt_idx[0]

    # restore optimal decision
    opt_part = np.zeros((K_gr,))
    opt_part[0] = opt_pals[K_gr - 1, 0]  # it was just "Kgr, 0"

    K_iter = K_gr - 1
    for i in range(0, K_iter):
        opt_part[i + 1] = opt_pals[K_iter - i - 1, int(opt_part[i])]
    return Q, opt_part


def my_qu_ix_w(invec, yinwec):
    PAR = 1
    PAR_sig_min = 0.1
    if (invec[-1] - invec[0]) <= PAR_sig_min or np.sum(yinwec) <= 1.0e-3:
        wyn = np.inf
    else:
        wwec = yinwec / (np.sum(yinwec))
        wyn = (PAR + np.sqrt(np.sum(((invec - np.sum(invec * wwec)) ** 2) * wwec))) / (
            np.max(invec) - np.min(invec)
        )  # lots of matrix operations
    return wyn


def normal_pdf(x, mu, sigma):
    return np.divide(
        np.exp(-0.5 * np.power(np.divide(x - mu, sigma), 2)), 2.506628274631 * sigma
    )


def plot_bic(BIC, index, path_visualisation_name):
    bic_arr = np.array(BIC)
    # idx = np.unravel_index(bic_arr.argmin(), bic_arr.shape)[0]
    fig, ax = plt.subplots(1, figsize=(10, 10))
    color = ["#56A8CBFF" for _ in range(len(BIC))]
    color[index] = "#FF4F58FF"
    sns.barplot([j for j in range(2, len(BIC) + 2)], BIC, palette=color, ax=ax)
    fig.savefig(path_visualisation_name)
    plt.close(fig)
