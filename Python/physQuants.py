import numpy as np
import mpi_functions as mpi_fncs

# E=sqrt(m^2+Q^2)

def energy( mEff, Qsq, L ):

    return np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * Qsq )


# E+m

def Epm( mEff, Qsq, L ):

    return energy( mEff, Qsq, L ) + mEff


# E-m

def Emm( mEff, Qsq, L ):

    return energy( mEff, Qsq, L ) - mEff


# 2E+m

def twoEpm( mEff, Qsq, L ):

    return 2* energy( mEff, Qsq, L ) + mEff


# 2E-m

def twoEmm( mEff, Qsq, L ):

    return 2 * energy( mEff, Qsq, L ) - mEff


# KK=sqrt(2E(E+m))

def KK( mEff, Qsq, L ):

    return np.sqrt( 2.0 * energy( mEff, Qsq, L ) \
                    * ( energy( mEff, Qsq, L ) \
                        + mEff ) )


# C1=sqrt(m/E)

def C1( mEff, Qsq, L ):

    return np.sqrt( mEff / energy( mEff, Qsq, L ) )


# C2=m/sqrt(EE')

def C2( mEff, Qsq, Qsq_prime, L ):

    return mEff / np.sqrt( energy( mEff, Qsq, L ) \
                           * energy( mEff, Qsq_prime, L ) )


def formFactorKinematic( ratio_err, mEff, Q, L, particle, formFactor ):

    # ratio_err[ Q, r ]
    # mEff[ b ]
    # momList[ Q, p ]
  
    momNum = ratio_err.shape[ 0 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    assert len( Q ) == momNum, "Error (kineFactor): " \
        + "momentum dimension of ratio errors " \
        + str( momNum ) + " and momentum transfers " \
        + str( len( Q ) ) + " do not match. "

    # kineFactor[ b, Q, r, [GE,GM] ]

    kineFactor = np.zeros( ( binNum, momNum, ratioNum, 2 ) )

    for b in range( binNum ):

        for q in range( momNum ):

            Qsq = np.dot( Q[ q ], Q[ q ] )

            if formFactor == "EM":

                #if particle == "nucleon":

                # R_P0mu0 = [ (E+m), 0 ]
                # R_P0mu1 = [ -Q_x, 0 ]
                # R_P0mu2 = [ -Q_y, 0 ]
                # R_P0mu3 = [ -Q_z, 0 ]
                # R_P4mu2 = [ 0, -Q_z ]
                # R_P4mu3 = [ 0, Q_y ]
                # R_P5mu1 = [ 0, Q_z ]
                # R_P5mu3 = [ 0, -Q_x ]
                # R_P6mu1 = [ 0, -Q_y ]
                # R_P6mu2 = [ 0, Q_x ]
                
                kineFactor[ b, q ] = [ [ Epm( mEff[ b ], Qsq, L ), \
                                         0 ], \
                                       [ -2.0 * np.pi / L * Q[ q, 0 ], 0 ], \
                                       [ -2.0 * np.pi / L * Q[ q, 1 ], 0 ], \
                                       [ -2.0 * np.pi / L * Q[ q, 2 ], 0 ], \
                                       [ 0, -2.0 * np.pi / L * Q[ q, 2 ] ], \
                                       [ 0, 2.0 * np.pi / L * Q[ q, 1 ] ], \
                                       [ 0, 2.0 * np.pi / L * Q[ q, 2 ] ], \
                                       [ 0, -2.0 * np.pi / L * Q[ q, 0 ] ], \
                                       [ 0, -2.0 * np.pi / L * Q[ q, 1 ] ], \
                                       [ 0, 2.0 * np.pi / L * Q[ q, 0 ] ] ] \
                    / np.repeat( ratio_err[ q ], 2 ).reshape( ratioNum, 2 ) \
                    / KK( mEff[ b ], Qsq, L )
                    
                #else: 

                    # CJL: I do not know this one

            elif formFactor == "1D":
                
                #if particle == "nucleon":
                    
                    # CJL: I do not know this one

                #else:

                # {R_g0D0} = [ -1/4(E+m)(2E+m), -(E-m)(2E-m) ]
                # {R_g0Dx} = [ -i/2(E+m)Q_x, -2i(E-m)Q_x ]
                # {R_g0Dy} = [ -i/2(E+m)Q_y, -2i(E-m)Q_y ]
                # {R_g0Dz} = [ -i/2(E+m)Q_z, -2i(E-m)Q_z ]
                # {R_gxDy} = [ 1/2 Q_x Q_y, 2 Q_x Q_y ]
                # {R_gxDz} = [ 1/2 Q_x Q_z, 2 Q_x Q_z ]
                # {R_gyDz} = [ 1/2 Q_y Q_z, 2 Q_y Q_z ]

                kineFactor[ b, q ] = [ [ -0.25 * Epm( mEff[ b ], Qsq, L ) \
                                         * twoEpm( mEff[b ], Qsq, L ), \
                                         -Emm( mEff[ b ], Qsq, L ) \
                                         * twoEmm( mEff[ b ], Qsq, L ) ], \
                                       [ -0.5 * Epm( mEff[ b ], Qsq, L ) \
                                         * 2 * np.pi / L * Q[ q, 0 ], \
                                         -2 * Emm( mEff[ b ], Qsq, L )
                                         * 2 * np.pi / L * Q[ q, 0 ] ], \
                                       [ -0.5 * Epm( mEff[ b ], Qsq, L ) \
                                         * 2 * np.pi / L * Q[ q, 1 ], \
                                         -2 * Emm( mEff[ b ], Qsq, L )
                                         * 2 * np.pi / L * Q[ q, 1 ] ], \
                                       [ -0.5 * Epm( mEff[ b ], Qsq, L ) \
                                         * 2 * np.pi / L * Q[ q, 2 ], \
                                         -2 * Emm( mEff[ b ], Qsq, L )
                                         * 2 * np.pi / L * Q[ q, 2 ] ], \
                                       [ 0.5 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 0 ] * Q[ q, 1 ], \
                                         2 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 0 ] * Q[ q, 1 ] ], \
                                       [ 0.5 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 0 ] * Q[ q, 2 ], \
                                         2 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 0 ] * Q[ q, 2 ] ],  \
                                       [ 0.5 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 1 ] * Q[ q, 2 ], \
                                         2 * ( 2 * np.pi / L ) ** 2 \
                                         * Q[ q, 1 ] * Q[ q, 2 ] ] ] \
                    / np.repeat( ratio_err[ q ], 2 ).reshape( ratioNum, 2 ) \
                    * C1( mEff[ b ], Qsq, L )

    return kineFactor


def decompFormFactors( decomp, ratio, ratio_err, Qsq_start, Qsq_end ):

    binNum = decomp.shape[ 0 ]

    A = np.zeros( ( binNum ) )
    B = np.zeros( ( binNum ) )

    for b in range( binNum ):

        A[ b ] = np.sum( decomp[ b, ..., 0 ] \
                                * ratio[ b, \
                                         Qsq_start \
                                         : Qsq_end + 1 ] \
                                / ratio_err[ Qsq_start \
                                             : Qsq_end \
                                             + 1 ] )

        B[ b ] = np.sum( decomp[ b, ..., 1 ] \
                                * ratio[ b, \
                                         Qsq_start \
                                         : Qsq_end + 1 ] \
                                / ratio_err[ Qsq_start \
                                             : Qsq_end \
                                             + 1 ] )
    
    return A, B


def calc_GE_GM( gE, gM, mEff, Qsq, L ):

    # gE[ b ], gM[ b ], mEff[ b ]
    # Qsq, L

    GE = gE \
         + ( energy( mEff, Qsq, L ) - mEff ) \
         / ( energy( mEff, Qsq, L ) + mEff ) * gM

    GM = 2 * mEff / ( energy( mEff, Qsq, L ) + mEff ) \
         * ( gM - gE )

    return GE, GM


# Convert Q^2 from units of (2pi/L)^2 to GeV^2

# Qsq: Q^2 values to be converted
# mEff: Effective mass of particle
# a: Lattice spacing of ensemble
# L: Spacial dimension length of ensemble

def convertQsqToGeV( Qsq, mEff, a, L ):

    Qsq_GeV = 2.0 * ( 0.197 / a ) ** 2 * mEff * ( energy( mEff, Qsq, L ) \
                                                  - mEff )

    return Qsq_GeV


# Calcuate the effective mass from two-point functions which have been
# symmetrized

# twop: Symmetrized two-point functions with last dimension as time

def mEffFromSymTwop( twop ):

    halfT = twop.shape[ -1 ]

    mEff = np.zeros( twop.shape )

    for t in range( 1, halfT - 1 ):

        mEff[ ..., t ] = 1.0 / 2.0 \
                         * np.log( ( twop[ ..., t - 1 ] \
                                     + np.sqrt(twop[ ..., \
                                                     t - 1 ] ** 2 \
                                               - twop[ ..., \
                                                       halfT - 1 ] ** 2) ) \
                                   / ( twop[ ..., t + 1 ] \
                                       + np.sqrt(twop[ ..., \
                                                       t + 1 ] ** 2 \
                                                 - twop[ ..., \
                                                         halfT - 1 ] ** 2) ))

    return mEff


# Calculate the effective mass from two-point functions

# twop: Two-point functions with last dimension as time

def mEff( twop ):

    mEff = np.zeros( twop.shape )

    # Loop through timestep, excluding the last timestep

    for t in range( twop.shape[ -1 ] - 1 ):

        mEff[ ..., t ] = np.log( twop[ ..., t ] / twop[ ..., t + 1 ] )

    # Calculate effective mass at last timestep, 
    # applying boundary conditions

    mEff[ ..., -1 ] = np.log( twop[ ..., -1 ] / twop[ ..., 0 ] )

    return mEff


# Calculate the quark momentum fraction <x> for three-point functions with
# zero final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle

def calcAvgX( threep, twop_tsink, mEff ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):
           
        avgX[ :, t ] = -4.0 / 3.0 / mEff * threep[ :, t ] / twop_tsink

    return avgX


# Calculate the quark momentum fraction <x> for three-point functions with
# finite final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle
# momSq: Final momentum squared
# L: Spacial dimension length of ensemble

def calcAvgX_momBoost( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    # prefactor = 8/3 * E / ( E^2 + p^2 )

    preFactor = -8.0 / 3.0 * energy( mEff, momSq, L ) \
                / ( energy( mEff, momSq, L ) ** 2 \
                    + ( 2.0 * np.pi / L ) ** 2 * momSq )

    #preFactor = 1.0

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


# Calculate the axial charge gA.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink

def calcgA( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    gA = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        gA[ :, t ] = threep[ :, t ] / twop_tsink

    return gA


# Calculate the cosh form of the two-point functions

# t: 
# energy:
# tsink:

def twopCosh( t, energy, tsink ):

    return np.exp( - energy * t ) + np.exp( - energy * ( tsink - t ) )


def calcRatio_Q( threep, twop, tsink ):
    
    # threep[ ..., Q, t ]
    # twop[ ..., Q, t ]

    ratio = np.zeros( threep.shape )
    
    for q in range( threep.shape[ -2 ] ):
        for t in range( threep.shape[ -1 ] ):

            ratio[..., q, t] = threep[ ..., q, t ] / twop[ ..., 0, tsink ] \
                               * np.sqrt( np.abs( twop[ ..., q, tsink - t ] \
                                                  * twop[ ..., 0, t ] \
                                                  * twop[ ..., 0, tsink ] \
                                                  / ( twop[ ..., 0, tsink - t ] \
                                                      * twop[ ..., q, t ] \
                                                      * twop[ ..., q, tsink ] ) ) )  

    return ratio

# Calculate the electromagnetic form factor.

# threep:

def calcEMFF( threep, twop, Qsq, mEff, tsink, L ):

    # threep[ q, b, t ]
    # twopp[ q, b, t ]
    # Qsq[ q ]
    # mEff[ b ]
    # tsink
    # L

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        factor = 0.25 * np.sqrt( energy( mEff, Qsq[ q ], L ) / mEff ) \
                 / ( energy( mEff, Qsq[ q ], L ) + mEff )

        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] \
                              / twop[ 0, :, tsink ] \
                              * np.sqrt( twop[ q, :, tsink - t ] \
                                         * twop[ 0, :, t ] \
                                         * twop[ 0, :, tsink ] \
                                         / ( twop[ 0, :, tsink - t ] \
                                             * twop[ q, :, t ] \
                                             * twop[ q, :, tsink ] ) )

    return emff


def calcEMFF_cosh( threep, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        factor = np.sqrt( energy / mEff ) * energy / ( energy + mEff )

        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / threep[ 0, :, t ] \
                              * twopCosh( t, mEff, tsink ) / twopCosh( t, energy, tsink )

    return emff


def calcEMFF_ratio( threep, twop, Qsq, mEff ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        for b in range( threep.shape[ 1 ] ):

            for t in range( threep.shape[ 2 ] ):

                emff[ q, b, t ] = threep[ q, b, t ] / twop[ q, b,  t ] \
                                   / threep[0,b,t] * twop[ 0, b, t ]

    return emff

