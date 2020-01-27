import tensorflow as tf
import Lorenz_Vector_tf as lv
import form_factors
import math

cos = tf.math.cos
sin = tf.math.sin
sqrt = tf.math.sqrt
arccos = tf.math.acos
T = tf.transpose
pi = tf.constant(math.pi)

class BHDVCS(object):

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998) # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272) #Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379*1000000) # Conversion from GeV to NanoBarn
        # Elastic FF
        # self.F1  # Dirac FF - helicity conserving (non spin flip)
        # self.F2  # Pauli FF - helicity non-conserving (spin flip)

        # self.QQ, x, t, k# creo q no hace falta
        # self.y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau

        # 4-momentum vectors
        self.K = lv.LorentzVector()  
        self.KP = lv.LorentzVector()
        self.Q = lv.LorentzVector()
        self.QP  = lv.LorentzVector() 
        self.D  = lv.LorentzVector()
        self.p  = lv.LorentzVector()
        self.P  = lv.LorentzVector()
        # 4 - vector products independent of phi
        self.kkp = tf.constant(0.)
        self.kq =tf.constant(0.0)
        self.kp =tf.constant(0.)
        self.kpp=tf.constant(0.)
        # 4 - vector products dependent of phi
        # self.kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd

        #     #self.KK_T, KQP_T, KKP_T, KXQP_T, KD_T, DD_T
        # self.kk_T, kqp_T, kkp_T, kd_T, dd_T

        # self.s     # Mandelstam variable s which is the center of mass energy
        # self.Gamma # Factor in front of the cross section
        # self.jcob  #Defurne's Jacobian

        # self.AUUBH, BUUBH # Coefficients of the BH unpolarized structure function FUU_BH
        # self.AUUI, BUUI, CUUI # Coefficients of the BHDVCS interference unpolarized structure function FUU_I
        # self.con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI  # Coefficients times the conversion to nb and the jacobian
        # self.bhAUU, bhBUU # Auu and Buu term of the BH cross section
        # self.iAUU, iBUU, iCUU # Terms of the interference containing AUUI, BUUI and CUUI
        # self.xbhUU # Unpolarized BH cross section
        # self.xIUU # Unpolarized interference cross section

    def TProduct(self, v1, v2):
        tv1v2 = tf.constant(0.0)
        tv1v2 = v1.Px() * v2.Px() + v1.Py() * v2.Py()
        return tv1v2



    def SetKinematics(self, _QQ, _x, _t, _k):

        self.QQ = tf.constant(_QQ, dtype=tf.float32) #Q^2 value
        self.x = tf.constant(_x, dtype=tf.float32)   #Bjorken x
        self.t = tf.constant(_t, dtype=tf.float32)   #momentum transfer squared
        self.k = tf.constant(_k, dtype=tf.float32)   #Electron Beam Energy
        self.M2 = self.M*self.M #Mass of the proton  squared in GeV
        #fractional energy of virtual photon
        self.y = self.QQ / ( 2. * self.M * self.k * self.x ) # From eq. (23) where gamma is substituted from eq (12c)
        #squared gamma variable ratio of virtuality to energy of virtual photon
        self.gg = 4. * self.M2 * self.x * self.x / self.QQ # This is gamma^2 [from eq. (12c)]
        #ratio of longitudinal to transverse virtual photon flux
        self.e = ( 1 - self.y - ( self.y * self.y * (self.gg / 4.) ) ) / ( 1. - self.y + (self.y * self.y / 2.) + ( self.y * self.y * (self.gg / 4.) ) ) # epsilon eq. (32)
        #Skewness parameter
        self.xi = 1. * self.x * ( ( 1. + self.t / ( 2. * self.QQ ) ) / ( 2. - self.x + self.x * self.t / self.QQ ) ) # skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
        #Minimum t value
        self.tmin = ( self.QQ * ( 1. - sqrt( 1. + self.gg ) + self.gg / 2. ) ) / ( self.x * ( 1. - sqrt( 1. + self.gg ) + self.gg / ( 2.* self.x ) ) ) # minimum t eq. (29)
        #Final Lepton energy
        self.kpr = self.k * ( 1. - self.y ) # k' from eq. (23)
        #outgoing photon energy
        self.qp = self.t / 2. / self.M + self.k - self.kpr #q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
        #Final proton Energy
        self.po = self.M - self.t / 2. / self.M # This is p'_0 from eq. (28b)
        self.pmag = sqrt( ( -1*self.t ) * ( 1. - (self.t / (4. * self.M *self.M ))) ) # p' magnitude from eq. (28b)
        #Angular Kinematics of outgoing photon
        self.cth = -1. / sqrt( 1. + self.gg ) * ( 1. + self.gg / 2. * ( 1. + self.t / self.QQ ) / ( 1. + self.x * self.t / self.QQ ) ) # This is cos(theta) eq. (26)
        self.theta = arccos(self.cth) # theta angle
        #print('Theta: ', self.theta)
        #Lepton Angle Kinematics of initial lepton
        self.sthl = sqrt( self.gg ) / sqrt( 1. + self.gg ) * ( sqrt ( 1. - self.y - self.y * self.y * self.gg / 4. ) ) # sin(theta_l) from eq. (22a)
        self.cthl = -1. / sqrt( 1. + self.gg ) * ( 1. + self.y * self.gg / 2. )  # cos(theta_l) from eq. (22a)
        #ratio of momentum transfer to proton mass
        self.tau = -0.25 * self.t / self.M2

        # phi independent 4 - momenta vectors defined on eq. (21) -------------
        self.K.SetPxPyPzE( self.k * self.sthl, 0.0, self.k * self.cthl, self.k )
        self.KP.SetPxPyPzE( self.K[0], 0.0, self.k * ( self.cthl + self.y * sqrt( 1. + self.gg ) ), self.kpr )
        self.Q = self.K - self.KP
        self.p.SetPxPyPzE(0.0, 0.0, 0.0, self.M)

        # Sets the Mandelstam variable s which is the center of mass energy
        self.s = (self.p + self.K) * (self.p + self.K)

        # The Gamma factor in front of the cross section
        self.Gamma = 1. / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / self.PI / 16. / ( self.s - self.M2 ) / ( self.s - self.M2 ) / sqrt( 1. + self.gg ) / self.x

        # Defurne's Jacobian
        self.jcob = 1./ ( 2. * self.M * self.x * self.K[3] ) * 2. * self.PI * 2.
        #print("Jacobian: ", self.jcob)
        #___________________________________________________________________________________
    def Set4VectorsPhiDep(self, phi) :

        # phi dependent 4 - momenta vectors defined on eq. (21) -------------

        self.QP.SetPxPyPzE(self.qp * sin(self.theta) * cos( phi * self.RAD ), self.qp * sin(self.theta) * sin( phi * self.RAD ), self.qp * cos(self.theta), self.qp)
        self.D = self.Q - self.QP # delta vector eq. (12a)
        #print(self.D, "\n", self.Q, "\n", self.QP)
        self.pp = self.p + self.D # p' from eq. (21)
        self.P = self.p + self.pp
        self.P.SetPxPyPzE(.5*self.P.Px(), .5*self.P.Py(), .5*self.P.Pz(), .5*self.P.Pt())
        
        #____________________________________________________________________________________
    def Set4VectorProducts(self, phi) :

        # 4-vectors products (phi - independent)
        self.kkp  = self.K * self.KP   #(kk')
        self.kq   = self.K * self.Q    #(kq)
        self.kp   = self.K * self.p    #(pk)
        self.kpp  = self.KP * self.p   #(pk')

        #print('Four Vector Products:')
        #print(self.kkp)
        #print(self.kq)
        #print(self.kp)
        #print(self.kpp)

        # 4-vectors products (phi - dependent)
        self.kd   = self.K * self.D    #(kΔ)
        self.kpd  = self.KP * self.D   #(k'Δ)
        self.kP   = self.K * self.P    #(kP)
        self.kpP  = self.KP * self.P   #(k'P)
        self.kqp  = self.K * self.QP   #(kq')
        self.kpqp = self.KP * self.QP  #(k'q')
        self.dd   = self.D * self.D    #(ΔΔ)
        self.Pq   = self.P * self.Q    #(Pq)
        self.Pqp  = self.P * self.QP   #(Pq')
        self.qd   = self.Q * self.D    #(qΔ)
        self.qpd  = self.QP * self.D   #(q'Δ)

        # #Transverse vector products defined after eq.(241c) -----------------------
        self.kk_T = 0.5 * ( self.e / ( 1 - self.e ) ) * self.QQ  #
        self.kkp_T = self.kk_T  #
        self.kqp_T = ( self.QQ / ( sqrt( self.gg ) * sqrt( 1 + self.gg ) ) ) * sqrt ( (0.5 * self.e) / ( 1 - self.e ) ) * ( 1. + self.x * self.t / self.QQ ) * sin(self.theta) * cos( phi * self.RAD )
        self.kd_T = -1.* self.kqp_T
        self.dd_T = ( 1. - self.xi * self.xi ) * ( self.tmin - self.t )

        # kk_T = TProduct(K,K)
        # kkp_T = kk_T
        # kqp_T = TProduct(K,QP)
        # kd_T = -1.* kqp_T
        # dd_T = TProduct(D,D)
        
        #____________________________________________________________________________________
    def GetBHUUxs(self, F1, F2) :

        # Coefficients of the BH unpolarized structure function FUUBH
        self.AUUBH = ( (8. * self.M2) / (self.t * self.kqp * self.kpqp) ) * ( (4. * self.tau * (self.kP * self.kP + self.kpP * self.kpP) ) - ( (self.tau + 1.) * (self.kd * self.kd + self.kpd * self.kpd) ) )
        self.AUUBH = ( (8. * self.M2) / (self.t * self.kqp * self.kpqp) ) * ( (4. * self.tau * (self.kP * self.kP + self.kpP * self.kpP) ) - ( (self.tau + 1.) * (self.kd * self.kd + self.kpd * self.kpd) ) )
        self.BUUBH = ( (16. * self.M2) / (self.t* self.kqp * self.kpqp) ) * (self.kd * self.kd + self.kpd * self.kpd)

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        # I multiply by 2 because I think Auu and Buu are missing a factor 2
        self.con_AUUBH = 2. * self.AUUBH * self.GeV2nb * self.jcob
        self.con_BUUBH = 2. * self.BUUBH * self.GeV2nb * self.jcob

        # Unpolarized Coefficients multiplied by the Form Factors
        self.bhAUU = (self.Gamma/self.t) * self.con_AUUBH * ( F1 * F1 + self.tau * F2 * F2 )
        self.bhBUU = (self.Gamma/self.t) * self.con_BUUBH * ( self.tau * ( F1 + F2 ) * ( F1 + F2 ) ) 

        # Unpolarized BH cross section
        self.xbhUU = self.bhAUU + self.bhBUU

        return self.xbhUU
        
        #____________________________________________________________________________________
    def GetIUUxs(self, phi, F1, F2, ReH, ReE, ReHtilde) :

        # Interference coefficients given on eq. (241a,b,c)--------------------
        self.AUUI = -4.0 * cos( phi * self.RAD ) / (self.kqp * self.kpqp) * ( ( self.QQ + self.t ) * ( 2.0 * ( self.kP + self.kpP ) * self.kk_T   + ( self.Pq * self.kqp_T ) + 2.* ( self.kpP * self.kqp ) - 2.* ( self.kP * self.kpqp ) ) +
                                                        ( self.QQ - self.t + 4.* self.kd ) * self.Pqp * ( self.kkp_T + self.kqp_T - 2.* self.kkp ) )
        self.BUUI = 2.0 * self.xi * cos( phi * self.RAD ) / ( self.kqp * self.kpqp) * ( ( self.QQ + self.t ) * ( 2.* self.kk_T * ( self.kd + self.kpd ) + self.kqp_T * ( self.qd - self.kqp - self.kpqp + 2.*self.kkp ) + 2.* self.kqp * self.kpd - 2.* self.kpqp * self.kd ) +
                                                                ( self.QQ - self.t + 4.* self.kd ) * ( ( self.kk_T - 2.* self.kkp ) * self.qpd - self.kkp * self.dd_T - 2.* self.kd_T * self.kqp ) ) / self.tau
        self.CUUI = 2.0 * cos( phi * self.RAD ) / ( self.kqp * self.kpqp) * ( -1. * ( self.QQ + self.t ) * ( 2.* self.kkp - self.kpqp - self.kqp ) * self.kd_T + ( self.QQ - self.t + 4.* self.kd ) * ( ( self.kqp + self.kpqp ) * self.kd_T + self.dd_T * self.kkp ) )

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        self.con_AUUI = self.AUUI * self.GeV2nb * self.jcob
        self.con_BUUI = self.BUUI * self.GeV2nb * self.jcob
        self.con_CUUI = self.CUUI * self.GeV2nb * self.jcob

        #Unpolarized Coefficients multiplied by the Form Factors
        self.iAUU = (self.Gamma/(-1*self.t * self.QQ)) * self.con_AUUI * ( F1 * ReH + self.tau * F2 * ReE )
        self.iBUU = (self.Gamma/(-1*self.t * self.QQ)) * self.con_BUUI * self.tau * ( F1 + F2 ) * ( ReH + ReE )
        self.iCUU = (self.Gamma/(-1*self.t * self.QQ)) * self.con_CUUI * ( F1 + F2 ) * ReHtilde

        # Unpolarized BH-DVCS interference cross section
        self.xIUU = self.iAUU + self.iBUU + self.iCUU

        return self.xIUU
    
    def TotalUUXS(self, angle, par):
        phi = angle[0]
        #print(phi)
        converter = tf.constant(self.PI/180)
        phi_2 = phi*converter
        phi_3 = phi*self.PI/180.0
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics( par[0], par[1], par[2], par[3] )
        self.Set4VectorsPhiDep(phi)
        self.Set4VectorProducts(phi)

        xsbhuu	 = self.GetBHUUxs(par[4], par[5])
        xsiuu	 = self.GetIUUxs(phi, par[4], par[5], par[6], par[7], par[8])
        
        tot_sigma_uu = xsbhuu + xsiuu + par[9] # Constant added to account for DVCS contribution
        #print(phi)
        #print(phi_2)
        #rint(phi_3)
        #print(par[0], ' ', par[1], ' ', par[2], ' ', par[3], ' ' ,par[4], ' ', par[5], ' ', par[6], ' ', par[7], ' ', par[8])
        #print('xsbhuu: ', xsbhuu)
        #print('xsiuu: ', xsiuu)
        #print(xsbhuu, " ", xsiuu, " ", tot_sigma_uu)
        return tot_sigma_uu
    
    def TotalUUXS_curve_fit(self, data, par1, par2, par3):
        phi, kin1, kin2, kin3, kin4, const = data
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics( kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(phi)
        self.Set4VectorProducts(phi)

        xsbhuu	 = self.GetBHUUxs(par1, par2)
        xsiuu	 = self.GetIUUxs(phi, par1, par2, par1, par2, par3)
        

        tot_sigma_uu = xsbhuu + xsiuu +  const# Constant added to account for DVCS contribution
        return tot_sigma_uu