import math

import tensorflow as tf

from kelly import kelly
from fourvector import *
import graphtrace
import form_factors

cos = tf.math.cos
sin = tf.math.sin
sqrt = tf.math.sqrt
T = tf.transpose
pi = tf.constant(math.pi)

# mass of the proton in GeV
M = tf.constant(0.93828)

# electromagnetic fine structure constant
alpha = tf.constant(0.00729927007)

# conversion from GeV to NanoBarn
conversion = tf.constant(0.389379e7)

def fake_ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    one = phi * reH + xbj * imH + Q2 * reE + imE + k0 * reHt + imHt + (1 - reEt) + imEt
    return one


def easy_loss(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    return reH


def fake_switch_ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    one = cos(phi) * reH + xbj * imH + Q2 * reE + imE + k0 * reHt + imHt + (1 - reEt) + imEt
    two = phi * reH**2 + xbj * imH + 1 + Q2 * reE + imE + k0 * reHt + imHt + (1 - reEt) + imEt
    three = sin(phi) * reH + xbj * imH + Q2 * reE + imE + k0 * 2.*reHt + imHt + (1 - reEt) + imEt
    four = cos(phi) * (reH + xbj * imH - Q2) * reE + imE + k0 * reHt + 3*imHt + (1 - reEt) + imEt
    five = phi * reH + xbj * 19 * imH + Q2 * (reE + imE + k0) * reHt + imHt + (1 - reEt) + imEt
    six = 10*phi * reH + xbj * imH + Q2 * reE + imE + k0 * reHt + imHt + (1 + reEt) + imEt
    seven = phi/2 * reH + xbj * imH + Q2 * reE + imE + k0 * reHt - imHt + (1 + reEt) + imEt
    sigmas = T(tf.stack([one, two, three, four, five, six, seven]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return tf.gather_nd(sigmas, gather_nd_idxs)

   


@graphtrace.trace_graph
def ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0):
    """ Calculation of cross sections from form factor predictions.

    Autoformatted with black - which makes it much harder to follow but much easier
    to compare with the original paper.

    Args:
        reH (tf.Tensor) : Tensor of shape (batch_size,). 0 index of model output. dtype tf.float32
        imH (tf.Tensor) : Tensor of shape (batch_size,). 1 index of model output. dtype tf.float32
        reE (tf.Tensor) : Tensor of shape (batch_size,). 2 index of model output. dtype tf.float32
        imE (tf.Tensor) : Tensor of shape (batch_size,). 3 index of model output. dtype tf.float32
        reHt (tf.Tensor) : Tensor of shape (batch_size,). 4 index of model output. dtype tf.float32
        imHt (tf.Tensor) : Tensor of shape (batch_size,). 5 index of model output. dtype tf.float32
        reEt (tf.Tensor) : Tensor of shape (batch_size,). 6 index of model output. dtype tf.float32
        imEt (tf.Tensor) : Tensor of shape (batch_size,). 7 index of model output. dtype tf.float32
        xbj (tf.Tensor) : Tensor of shape (batch_size,). 0 index of kinematic input. dtype tf.float32
        t (tf.Tensor) : Tensor of shape (batch_size,). 1 index of kinematic input. dtype tf.float32
        Q2 (tf.Tensor) : Tensor of shape (batch_size,). 2 index of kinematic input. dtype tf.float32
        phi (tf.Tensor) : Tensor of shape (batch_size,). 3 index of kinematic input. dtype tf.float32
        L (tf.Tensor) : Tensor of shape (batch_size,). 0 index of sigma_true label. dtype *tf.int32*
    
    Returns:
        Calculated cross section tf.Tensor of shape (batch_size,)
    """
    #print(reH.shape)
    depth_vector = tf.ones((reH.shape[0],), dtype=tf.float32)

    ###################################
    ## Secondary Kinematic Variables ##
    ###################################

    # energy of the virtual photon
    tmp = (2.0 * M * xbj)
    nu = Q2 / tmp

    # skewness parameter set by xbj, t, and Q^2
    xi = xbj * ((1.0 + (t / (2.0 * Q2))) / (2.0 - xbj + ((xbj * t) / Q2)))

    # gamma variable ratio of virtuality to energy of virtual photon
    gamma = sqrt(Q2) / nu

    # fractional energy of virtual photon
    y = sqrt(Q2) / (gamma * k0)

    # final lepton energy
    k0p = k0 * (1.0 - y)

    # minimum t value
    t0 = -(4.0 * xi * xi * M * M) / (1.0 - (xi * xi))

    # Lepton Angle Kinematics of initial lepton
    costl = -(1.0 / (sqrt(1.0 + gamma * gamma))) * (1.0 + (y * gamma * gamma / 2.0))
    sintl = (gamma / (sqrt(1.0 + gamma * gamma))) * sqrt(
        1.0 - y - (y * y * gamma * gamma / 4.0)
    )

    # Lepton Angle Kinematics of final lepton
    sintlp = sintl / (1.0 - y)
    costlp = (costl + y * sqrt(1.0 + gamma * gamma)) / (1.0 - y)

    # final proton energy
    p0p = M - (t / (2.0 * M))

    # ratio of longitudinal to transverse virtual photon flux
    eps = (1.0 - y - 0.25 * y * y * gamma * gamma) / (
        1.0 - y + 0.5 * y * y + 0.25 * y * y * gamma * gamma
    )

    # angular kinematics of outgoing photon
    cost = -(1 / (sqrt(1 + gamma * gamma))) * (
        1 + (0.5 * gamma * gamma) * ((1 + (t / Q2)) / (1 + ((xbj * t) / (Q2))))
    )
    sint = sqrt(1.0 - cost * cost)

    # outgoing photon energy
    q0p = (sqrt(Q2) / gamma) * (1 + ((xbj * t) / Q2))

    # ratio of momentum transfer to proton mass
    tau = -t / (4.0 * M * M)

    ###############################################################################
    ## Creates arrays of 4-vector kinematics uses in Bethe Heitler Cross Section ##
    ###############################################################################

    # initial proton 4-momentum
    p = T(
        tf.convert_to_tensor(
            [
                M * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
                0.0 * depth_vector,
            ]
        )
    )

    # initial lepton 4-momentum
    k = T(
        tf.convert_to_tensor(
            [k0 * depth_vector, k0 * sintl, 0.0 * depth_vector, k0 * costl]
        )
    )

    # final lepton 4-momentum
    kp = T(
        tf.convert_to_tensor(
            [k0p * depth_vector, k0p * sintlp, 0.0 * depth_vector, k0p * costlp]
        )
    )

    # virtual photon 4-momentum
    q = k - kp

    ##################################
    ## Creates four vector products ##
    ##################################
    plp = product(p, p)  # pp
    qq = product(q, q)  # qq
    kk = product(k, k)  # kk
    kkp = product(k, kp)  # kk'
    kq = product(k, q)  # kq
    pk = product(k, p)  # pk
    pkp = product(kp, p)  # pk'

    # sets the Mandelstam variables s which is the center of mass energy
    s = product(k, k) + 2 * product(k, p) + product(p, p)

    # the Gamma factor in front of the cross section
    Gamma = (alpha ** 3) / (
        16.0 * (pi ** 2) * ((s - M * M) ** 2) * sqrt(1.0 + gamma ** 2) * xbj
    )

    phi = phi * 0.0174532951  # radian conversion

    # final real photon 4-momentum
    qp = T(
        tf.convert_to_tensor(
            [
                q0p * depth_vector,
                q0p * sint * T(cos(phi)),
                q0p * sint * T(sin(phi)),
                q0p * cost * depth_vector,
            ]
        )
    )

    # momentum transfer Δ from the initial proton to the final proton
    d = q - qp

    # final proton momentum
    pp = p + d

    # average initial proton momentum
    P = 0.5 * (p + pp)

    # 4-vector products of variables multiplied by spin vectors
    ppSL = ((M) / (sqrt(1.0 + gamma ** 2))) * (
        xbj * (1.0 - (t / Q2)) - (t / (2.0 * M ** 2))
    )
    kSL = (
        ((Q2) / (sqrt(1.0 + gamma ** 2)))
        * (1.0 + 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )
    kpSL = (
        ((Q2) / (sqrt(1 + gamma ** 2)))
        * (1 - y - 0.5 * y * gamma ** 2)
        * (1.0 / (2.0 * M * xbj * y))
    )

    # 4-vector products denoted in the paper by the commented symbols
    kd = product(k, d)  # dΔ
    kpd = product(kp, d)  # k'Δ
    kP = product(k, P)  # kP
    kpP = product(kp, P)  # k'P
    kqp = product(k, qp)  # kq'
    kpqp = product(kp, qp)  # k'q'
    dd = product(d, d)  # ΔΔ
    Pq = product(P, q)  # Pq
    Pqp = product(P, qp)  # Pq'
    qd = product(q, d)  # qΔ
    qpd = product(qp, d)  # q'Δ

    # transverse vector products
    kkT = tproduct(k, k)
    kqpT = tproduct(k, qp)
    kkpT = tproduct(k, kp)
    ddT = tproduct(d, d)
    kdT = tproduct(k, d)
    kpqpT = tproduct(kp, qp)
    qpdT = tproduct(qp, d)
    kPT = tproduct(k, P)
    kpPT = tproduct(kp, P)
    qpPT = tproduct(qp, P)
    kpdT = tproduct(kp, d)

    # light cone variables expressed as A^{+-} = 1/sqrt(2)(A^{0} +- A^{3})
    inv_root_2 = 1 / sqrt(2.0)
    kplus = T(inv_root_2 * (k[..., 0] + k[..., 3]))
    kpplus = T(inv_root_2 * (kp[..., 0] + kp[..., 3]))
    kminus = T(inv_root_2 * (k[..., 0] - k[..., 3]))
    kpminus = T(inv_root_2 * (kp[..., 0] - kp[..., 3]))
    qplus = T(inv_root_2 * (q[..., 0] + q[..., 3]))
    qpplus = T(inv_root_2 * (qp[..., 0] + qp[..., 3]))
    qminus = T(inv_root_2 * (q[..., 0] - q[..., 3]))
    qpminus = T(inv_root_2 * (qp[..., 0] - qp[..., 3]))
    Pplus = T(inv_root_2 * (P[..., 0] + P[..., 3]))
    Pminus = T(inv_root_2 * (P[..., 0] - P[..., 3]))
    dplus = T(inv_root_2 * (d[..., 0] + d[..., 3]))
    dminus = T(inv_root_2 * (d[..., 0] - d[..., 3]))

    # expresssions used that appear in coefficient calculations
    Dplus = (1 / (2 * kpqp)) - (1 / (2 * kqp))
    Dminus = (1 / (2 * kpqp)) + (1 / (2 * kqp))

    # calculates BH
    AUUBH = ((8.0 * M * M) / (t * kqp * kpqp)) * (
        (4.0 * tau * (kP * kP + kpP * kpP)) - ((tau + 1.0) * (kd * kd + kpd * kpd))
    )
    BUUBH = ((16.0 * M * M) / (t * kqp * kpqp)) * (kd * kd + kpd * kpd)

    # calculates BHLL
    ALLBH = -((8.0 * M * M) / (t * kqp * kpqp)) * (
        (ppSL / M) * ((kpd * kpd - kd * kd) - 2.0 * tau * (kpd * pkp - kd * pk))
        + t * (kSL / M) * (1.0 + tau) * kd
        - t * (kpSL / M) * (1.0 + tau) * kpd
    )
    BLLBH = ((8.0 * M * M) / (t * kqp * kpqp)) * (
        (ppSL / M) * (kpd * kpd - kd * kd) + t * (kSL / M) * kd - t * (kpSL / M) * kpd
    )

    # converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBH = (Gamma / t) * AUUBH * conversion
    con_BUUBH = (Gamma / t) * BUUBH * conversion

    # converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBH = (Gamma / t) * ALLBH * conversion
    con_BLLBH = (Gamma / t) * BLLBH * conversion

    ffF1, ffF2, ffGM = kelly(-t)
    """
    ffF1 = form_factors.ffF1(-t)
    ffF2 = form_factors.ffF2(-t)
    ffGM = form_factors.ffGM(-t)
    """

    # unpolarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # we use the Galster Form Factors as approximations
    bhAUU = con_AUUBH * ((ffF1 * ffF1) + (tau * ffF2 * ffF2))
    bhBUU = con_BUUBH * (tau * ffGM * ffGM)

    # polarized Coefficients multiplied by the Form Factors calculated in form_factor.py
    # using the Galster Form Factor Model
    bhALL = con_ALLBH * (ffF2 * ffGM)
    bhBLL = con_BLLBH * (ffGM * ffGM)

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBH = bhAUU + bhBUU
    XSXLLBH = bhALL + bhBLL

    # Calculates the Unpolarized Coefficients in front of the Elastic Form Factors and
    # Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AUUBHDVCS = -16 * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpP
        + (2 * kpqp - 2 * kkpT - kpqpT) * kP
        + kpqp * kPT
        + kqp * kpPT
        - 2 * kkp * kPT
    ) * cos(phi) - 16 * Dminus * (
        (2 * kkp - kpqpT - kkpT) * Pqp + 2 * kkp * qpPT - kpqp * kPT - kqp * kpPT
    ) * cos(
        phi
    )
    BUUBHDVCS = -8 * xi * Dplus * (
        (kqpT - 2 * kkT - 2 * kqp) * kpd
        + (2 * kpqp - 2 * kkpT - kpqpT) * kd
        + kpqp * kdT
        + kqp * kpdT
        - 2 * kkp * kdT
    ) * cos(phi) - 8 * xi * Dminus * (
        (2 * kkp - kpqpT - kkpT) * qpd + 2 * kkp * qpdT - kpqp * kdT - kqp * kpdT
    ) * cos(
        phi
    )
    CUUBHDVCS = -8 * Dplus * (
        (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * cos(phi) - 8 * Dminus * (
        (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        + 2 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * cos(
        phi
    )

    # Calculates the Unpolarized Beam Polarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    AULBHDVCS = -16 * Dplus * (
        kpP * (2 * kkT - kqpT + 2 * kqp)
        + kP * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kPT
        - kpqp * kPT
        - kqp * kpPT
    ) * sin(phi) - 16 * Dminus * (
        Pqp * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )
    BULBHDVCS = -8 * xi * Dplus * (
        kpd * (2 * kkT - kqpT + 2 * kqp)
        + kd * (2 * kkpT - kpqpT + 2 * kpqp)
        + 2 * kkp * kdT
        - kpqp * kdT
        - kqp * kpdT
    ) * sin(phi) - 8 * xi * Dminus * (
        qpd * (kkpT + kpqpT - 2 * kkp) - (2 * kkp * qpdT - kpqp * kdT - kqp * kpdT)
    ) * sin(
        phi
    )
    CULBHDVCS = -8 * Dplus * (
        2 * (2 * kkp * kdT - kpqp * kdT - kqp * kpdT)
        + 4 * xi * (2 * kkp * kPT - kpqp * kPT - kqp * kpPT)
    ) * sin(phi) - 8 * Dminus * (
        -2 * (kkp * qpdT - kpqp * kdT - kqp * kpdT)
        - 4 * xi * (kkp * qpPT - kpqp * kPT - kqp * kpPT)
    ) * sin(
        phi
    )

    # Calculates the Polarized Beam Unpolarized Target Coefficients in front of the
    # Elastic Form Factors and Compton Form Factors
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALUBHDVCS = (
        16
        * Dplus
        * (
            2
            * (
                k[:, 1] * Pplus * kp[:, 1] * kminus
                - k[:, 1] * Pplus * kpminus * k[:, 1]
                + k[:, 1] * Pminus * kpplus * k[:, 1]
                - k[:, 1] * Pminus * kp[:, 1] * kplus
                + k[:, 1] * P[:, 1] * kpminus * kplus
                - k[:, 1] * P[:, 1] * kpplus * kminus
            )
            + kp[:, 1] * Pplus * qpminus * k[:, 1]
            - kp[:, 1] * Pplus * qp[:, 1] * kminus
            + kp[:, 1] * Pminus * qp[:, 1] * kplus
            - kp[:, 1] * Pminus * qpplus * k[:, 1]
            + kp[:, 1] * P[:, 1] * qpplus * kminus
            - kp[:, 1] * P[:, 1] * qpminus * kplus
            + k[:, 1] * Pplus * qpminus * kp[:, 1]
            - k[:, 1] * Pplus * qp[:, 1] * kpminus
            + k[:, 1] * Pminus * qp[:, 1] * kpplus
            - k[:, 1] * Pminus * qpplus * kp[:, 1]
            + k[:, 1] * P[:, 1] * qpplus * kpminus
            - k[:, 1] * P[:, 1] * qpminus * kpplus
            + 2 * (qpminus * Pplus - qpplus * Pminus) * kkp
        )
        * sin(phi)
    )
    + 16 * Dminus * (
        2 * (kminus * kpplus - kplus * kpminus) * Pqp
        + kpminus * kplus * qp[:, 1] * P[:, 1]
        + kpplus * k[:, 1] * qpminus * P[:, 1]
        + kp[:, 1] * kminus * qpplus * P[:, 1]
        - kpplus * kminus * qp[:, 1] * P[:, 1]
        - kp[:, 1] * kplus * qpminus * P[:, 1]
        - kpminus * k[:, 1] * qpplus * P[:, 1]
        + kpminus * kplus * qp[:, 2] * P[:, 2]
        - kpplus * kminus * qp[:, 2] * P[:, 2]
    ) * sin(phi)
    BLUBHDVCS = 8 * xi * Dplus * (
        2
        * (
            k[:, 1] * dplus * kp[:, 1] * kminus
            - k[:, 1] * dplus * kpminus * k[:, 1]
            + k[:, 1] * dminus * kpplus * k[:, 1]
            - k[:, 1] * dminus * kp[:, 1] * kplus
            + k[:, 1] * d[:, 1] * kpminus * kplus
            - k[:, 1] * d[:, 1] * kpplus * kminus
        )
        + kp[:, 1] * dplus * qpminus * k[:, 1]
        - kp[:, 1] * dplus * qp[:, 1] * kminus
        + kp[:, 1] * dminus * qp[:, 1] * kplus
        - kp[:, 1] * dminus * qpplus * k[:, 1]
        + kp[:, 1] * d[:, 1] * qpplus * kminus
        - kp[:, 1] * d[:, 1] * qpminus * kplus
        + k[:, 1] * dplus * qpminus * kp[:, 1]
        - k[:, 1] * dplus * qp[:, 1] * kpminus
        + k[:, 1] * dminus * qp[:, 1] * kpplus
        - k[:, 1] * dminus * qpplus * kp[:, 1]
        + k[:, 1] * d[:, 1] * qpplus * kpminus
        - k[:, 1] * d[:, 1] * qpminus * kpplus
        + 2 * (qpminus * dplus - qpplus * dminus) * kkp
    ) * sin(phi) + 8 * xi * Dminus * (
        2 * (kminus * kpplus - kplus * kpminus) * qpd
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 2] * d[:, 2]
        - kpplus * kminus * qp[:, 2] * d[:, 2]
    ) * sin(
        phi
    )
    CLUBHDVCS = -8 * Dplus * (
        2
        * (kp[:, 1] * kpminus * kplus * d[:, 1] - kp[:, 1] * kpplus * kminus * d[:, 1])
        + kp[:, 1] * qpminus * kplus * d[:, 1]
        - kp[:, 1] * qpplus * kminus * d[:, 1]
        + k[:, 1] * qpminus * kpplus * d[:, 1]
        - k[:, 1] * qpplus * kpminus * d[:, 1]
    ) * sin(phi) - 8 * Dminus * (
        -kpminus * k[:, 1] * qpplus * d[:, 1]
        + kpminus * kplus * qp[:, 1] * d[:, 1]
        + kpplus * k[:, 1] * qpminus * d[:, 1]
        - kpplus * kminus * qp[:, 1] * d[:, 1]
        + kp[:, 1] * kminus * qpplus * d[:, 1]
        - kp[:, 1] * kplus * qpminus * d[:, 1]
        - qp[:, 2] * d[:, 2] * (kpplus * kminus - kpminus * kplus)
    ) * sin(
        phi
    )

    # Calculates the Longitudinally Polarized Coefficients in front of the EFFs
    # No conversion factor to nano barn, nor the Gamma/t factor is included
    ALLBHDVCS = -16 * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * Pplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * Pminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * P[:, 1]
        + kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * Pplus
        + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * Pminus
        + kp[:, 1] * (qpplus * kminus - qpminus * kplus) * P[:, 1]
        + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * Pplus
        + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * Pminus
        + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * P[:, 1]
        - 2 * kkp * (qpplus * Pminus - qpminus * Pplus)
    ) * cos(phi) - 16 * Dminus * (
        2 * Pqp * (kpplus * kminus - kpminus * kplus)
        + P[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + P[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    BLLBHDVCS = -8 * xi * Dplus * (
        2 * kp[:, 1] * (kp[:, 1] * kminus - kpminus * k[:, 1]) * dplus
        + 2 * kp[:, 1] * (kpplus * k[:, 1] - kp[:, 1] * kplus) * dminus
        + 2 * kp[:, 1] * (kpminus * kplus - kpplus * kminus) * d[:, 1]
        + kp[:, 1] * (qpminus * k[:, 1] - qp[:, 1] * kminus) * dplus
        + kp[:, 1] * (qp[:, 1] * kplus - qpplus * k[:, 1]) * dminus
        + kp[:, 1] * d[:, 1] * (qpplus * kminus - qpminus * kplus)
        + k[:, 1] * (qpminus * kp[:, 1] - qp[:, 1] * kpminus) * dplus
        + k[:, 1] * (qp[:, 1] * kpplus - qpplus * kp[:, 1]) * dminus
        + k[:, 1] * (qpplus * kpminus - qpminus * kpplus) * d[:, 1]
        - 2 * kkp * (qpplus * dminus - qpminus * dplus)
    ) * cos(phi) - 8 * xi * Dminus * (
        2 * qpd * (kpplus * kminus - kpminus * kplus)
        + d[:, 2] * qp[:, 2] * (kpplus * kminus - kpminus * kplus)
        + d[:, 1]
        * (
            kp[:, 1] * kplus * qpminus
            - kp[:, 1] * kminus * qpplus
            + kpminus * k[:, 1] * qpplus
            - kpminus * kplus * qp[:, 1]
            + kpplus * kminus * qp[:, 1]
            - kpplus * k[:, 1] * qpminus
        )
    ) * cos(
        phi
    )
    CLLBHDVCS = 16 * Dplus * (
        2 * (k[:, 1] * kminus * kpplus * d[:, 1] - k[:, 1] * kpminus * kplus * d[:, 1])
        + kp[:, 1] * qpplus * kminus * d[:, 1]
        - kp[:, 1] * qpminus * kplus * d[:, 1]
        + k[:, 1] * qpplus * kpminus * d[:, 1]
        - k[:, 1] * qpminus * kpplus * d[:, 1]
    ) * cos(phi) + 16 * Dminus * (
        -d[:, 1]
        * (
            kpminus * kplus * qp[:, 1]
            - kpminus * k[:, 1] * qpplus
            + kpplus * k[:, 1] * qpminus
            - kpplus * kminus * qp[:, 1]
            + kp[:, 1] * kminus * qpplus
            - kp[:, 1] * kplus * qpminus
        )
        + qp[:, 2] * kpplus * kminus * d[:, 2]
        - qp[:, 2] * kpminus * kplus * d[:, 2]
    ) * cos(
        phi
    )

    # Converted Unpolarized Coefficients with the Gamma factor and in nano-barn
    con_AUUBHDVCS = (Gamma / (Q2 * -t)) * AUUBHDVCS * conversion
    con_BUUBHDVCS = (Gamma / (Q2 * -t)) * BUUBHDVCS * conversion
    con_CUUBHDVCS = (Gamma / (Q2 * -t)) * CUUBHDVCS * conversion

    # Converted Longitudinally Polarized Coefficients with the Gamma Factor and in nano-barn
    con_ALLBHDVCS = (Gamma / (Q2 * -t)) * ALLBHDVCS * conversion
    con_BLLBHDVCS = (Gamma / (Q2 * -t)) * BLLBHDVCS * conversion
    con_CLLBHDVCS = (Gamma / (Q2 * -t)) * CLLBHDVCS * conversion

    # Converted Longitudinally Polarized Beam Unpolarized Target Coefficients with
    # the Gamma Factor and in nano-barn
    con_ALUBHDVCS = (Gamma / (Q2 * -t)) * ALUBHDVCS * conversion
    con_BLUBHDVCS = (Gamma / (Q2 * -t)) * BLUBHDVCS * conversion
    con_CLUBHDVCS = (Gamma / (Q2 * -t)) * CLUBHDVCS * conversion

    # Converted Longitudinally Polarized Target Unpolarized Beam Coefficients with
    # the Gamma Factor and in nano-barn
    con_AULBHDVCS = (Gamma / (Q2 * -t)) * AULBHDVCS * conversion
    con_BULBHDVCS = (Gamma / (Q2 * -t)) * BULBHDVCS * conversion
    con_CULBHDVCS = (Gamma / (Q2 * -t)) * CULBHDVCS * conversion

    # Unpolarized Coefficients multiplied by the Form Factors
    bhdvcsAUU = con_AUUBHDVCS * (ffF1 * reH + tau * ffF2 * reE)
    bhdvcsBUU = con_BUUBHDVCS * (ffGM * (reH + reE))
    bhdvcsCUU = con_CUUBHDVCS * (ffGM * reHt)

    # Polarized Coefficients multiplied by the Form Factors
    bhdvcsALU = con_ALUBHDVCS * (ffF1 * imH + tau * ffF2 * imE)
    bhdvcsBLU = con_BLUBHDVCS * (ffGM * (imH + imE))
    bhdvcsCLU = con_CLUBHDVCS * (ffGM * imHt)

    # Unpolarized Beam Polarized Target Coefficients multiplied by the Form Factors
    bhdvcsAUL = con_AULBHDVCS * (ffF1 * imHt - xi * ffF1 * imEt + tau * ffF2 * imEt)
    bhdvcsBUL = con_BULBHDVCS * (ffGM * imHt)
    bhdvcsCUL = con_CULBHDVCS * (ffGM * (imH + imE))

    # Polarized Beam Unpolarized Target Coefficients multiplied by the Form Factors
    bhdvcsALL = con_ALLBHDVCS * (ffF1 * reHt - xi * ffF1 * reEt + tau * ffF2 * reEt)
    bhdvcsBLL = con_BLLBHDVCS * (ffGM * reHt)
    bhdvcsCLL = con_CLLBHDVCS * (ffGM * (reH + reE))

    # Calculation of the Total Unpolarized Bethe-Heitler Cross Section
    XSXUUBHDVCS = bhdvcsAUU + bhdvcsBUU + bhdvcsCUU
    XSXLLBHDVCS = bhdvcsALL + bhdvcsBLL + bhdvcsCLL
    XSXULBHDVCS = bhdvcsAUL + bhdvcsBUL + bhdvcsCUL
    XSXLUBHDVCS = bhdvcsALU + bhdvcsBLU + bhdvcsCLU

    FUUT = (
        (Gamma / (Q2))
        * conversion
        * (
            4
            * (
                (1 - xi * xi) * (reH * reH + imH * imH + reHt * reHt + imHt * imHt)
                + ((t0 - t) / (2 * M * M))
                * (
                    reE * reE
                    + imE * imE
                    + xi * xi * reEt * reEt
                    + xi * xi * imEt * imEt
                )
                - ((2 * xi * xi) / (1 - xi * xi))
                * (reH * reE + imH * imE + reHt * reEt + imHt * imEt)
            )
        )
    )

    XSXUU = XSXUUBHDVCS + XSXUUBH + FUUT
    XSXLU = XSXLUBHDVCS
    XSXUL = XSXULBHDVCS
    XSXLL = XSXLLBHDVCS + XSXLLBH
    XSXALU = XSXLU / (XSXUU + 1e-8)
    XSXAUL = XSXUL / (XSXUU + 1e-8)
    XSXALL = XSXLL / (XSXUU + 1e-8)
    sigmas = T(tf.stack([XSXUU, XSXLU, XSXUL, XSXLL, XSXALU, XSXAUL, XSXALL]))
    gather_nd_idxs = tf.stack(
        [tf.range(sigmas.shape[0], dtype=tf.int32), L - 1], axis=1
    )
    return XSXUU#tf.gather_nd(sigmas, gather_nd_idxs)
