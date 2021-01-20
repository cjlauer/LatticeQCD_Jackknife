import numpy as np
import lqcdjk_fitting as fit
import functions as fncs
import mpi_functions as mpi_fncs
from mpi4py import MPI

# E = sqrt( m^2 + p^2 )

def energy( mEff, pSq, L ):

    return np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * pSq )


# Q^2 = (p_f - p_i)^2 - (E_f - E_i)^2

def calcQsq( p_fin, q_list, mEff, L, mpi_info ):

    # p_fin[ p ]
    # q_list[ q ]
    # m[ b ]
    # L

    binNum = len( mEff )
    
    Qsq_list = [ [] for b in mEff ]
    Qsq_where = [ [] for b in mEff ]

    for m, ib in fncs.zipXandIndex( mEff ):

        # Qsq_p_q[ p, q ]

        Qsq_p_q = np.zeros( ( len( p_fin ), len( q_list ) ) )

        for p, ip in fncs.zipXandIndex( p_fin ):
            for q, iq in fncs.zipXandIndex( q_list ):

                p_ini = p - q
                
                Qsq_p_q[ ip, iq ] \
                    = ( 2. * np.pi / L ) ** 2 \
                    * np.dot( p - p_ini, p - p_ini ) \
                    - ( energy( m, np.dot( p, p ), L )
                        - energy( m, np.dot( p_ini, p_ini ), L ) ) ** 2

        Qsq_list[ ib ] = np.sort( np.unique( Qsq_p_q ) )

        Qsq_where[ ib ] = [ [] for qs in Qsq_list[ ib ] ]

        for qs, iqs in fncs.zipXandIndex( Qsq_list[ ib ] ):

            Qsq_where[ ib ][ iqs ] = Qsq_p_q == qs

    Qsq_list = np.array( Qsq_list )
    Qsq_where = np.array( Qsq_where )

    QsqNum = Qsq_list.shape[ -1 ]

    # Check that Q^2's are at the same place across bins

    for ib in range( 1, binNum ):
        
        if not np.array_equal( Qsq_where[ ib ],
                               Qsq_where[ 0 ] ):
            
            error = "Error (physQuants.fourVectorQsq): " \
                    "Qsq_where[ {} ] != Qsq_where[ 0 ]" 
        
            mpi_fncs.mpiPrint( Qsq_where[ ib ],
                               mpi_info )
            mpi_fncs.mpiPrint( Qsq_where[ 0 ],
                               mpi_info )

            mpi_fncs.mpiPrintError( error.format( ib ),
                                    mpi_info )

    Qsq_where = Qsq_where[ 0 ]

    return Qsq_list, QsqNum, Qsq_where


# KK = sqrt( 2E ( E + m ) )

def KK_nucleon( mEff, Qsq, L ):

    return np.sqrt( 2.0 * energy( mEff, Qsq, L ) \
                    * ( energy( mEff, Qsq, L ) \
                        + mEff ) )


# KK = C_1^-1 = 2 sqrt( EE' )

def KK_meson( mEff, pSq_ini, pSq_fin, L ):

    return 2.0 * np.sqrt( energy( mEff, pSq_ini, L )
                          * energy( mEff, pSq_fin, L ) )


def twopFit( c0, E0, t ):
#def twopFit( c0, c1, E0, E1, t ):

    return c0 * np.exp( -E0 * t )
    #return c0 * np.exp( -E0 * t ) + c1 * np.exp( -E1 * t )


def kineFactor( ratio_err, formFactor, particle, flavor,
                mEff, p_fin, Q, L, mpi_info ):

    if formFactor == "GE_GM":

        kineFactor = kineFactor_GE_GM( ratio_err, particle, flavor,
                                       mEff, p_fin, Q, L,
                                       mpi_info )

    elif formFactor == "A20_B20":

        kineFactor = kineFactor_A20_B20( ratio_err, particle, flavor,
                                         mEff, p_fin, Q, L,
                                         mpi_info )

    return kineFactor


def kineFactor_GE_GM( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                      mpi_info ):

    # ratio_err[ p, Q, r ]
    # "particle"
    # "flavor"
    # mEff[ b ]
    # p_fin[ p, pi ]
    # momList[ Q, qi ]
    # L

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                  p_fin.shape[ 0 ] ),
                           mpi_info )

    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                  Q.shape[ 0 ] ),
                           mpi_info )

    # kineFactor[ b, p, Q, ratio, [GE,GM] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                if particle == "nucleon":

                    Qsq = np.dot( q, q )
                    
                    kineFactor[ b, ip, iq ] \
                        = [ [ ( energy( mEff[ b ],
                                        Qsq, L ) \
                                + mEff[ b ] ), 0 ],
                            [ -2.0 * np.pi / L * q[ 0 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 1 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 2 ], 0 ],
                            [ 0, -2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 0 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 0 ] ] ] \
                        / np.repeat( ratio_err[ ip, iq ] ** 2,
                                     2).reshape( ratioNum, 2 ) \
                        / KK_nucleon( mEff[ b ], Qsq, L )
                    
                else: # particle == "meson"

                    p_ini = p - q

                    pSq_ini = np.dot( p_ini, p_ini )
                    pSq_fin = np.dot( p, p )

                    kineFactor[ b, ip, iq ] \
                        = [ [ energy( mEff[ b ],
                                      pSq_ini,
                                      L )
                              + energy( mEff[ b ],
                                        pSq_fin,
                                        L ),
                              0 ], 
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 0 ] + p[ 0 ] ),
                              0 ],
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 1 ] + p[ 1 ] ),
                              0 ],
                            [ -2.0 * np.pi / L
                              * ( p_ini[ 2 ] + p[ 2 ] ),
                              0 ] ] \
                        / np.repeat( ratio_err[ ip, iq ] ** 2,
                                     2).reshape( ratioNum, 2 ) \
                        / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                    
                # End if meson                                   
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


def kineFactor_A20_B20( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                        mpi_info ):

    # ratio_err[ p, Q, r ]
    # "particle"
    # "flavor"
    # mEff[ b ]
    # p_fin[ p, pi ]
    # momList[ Q, qi ]
    # L

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if particle == "nucleon":
        
        errorMessage = "Error (physQuants.kineFactor_A20_B20): " \
                       + "function not supported for nucleon"
        
        mpi_fncs.mpiPrintError( errorMessage,
                                mpi_info )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( finalMomentaNum,
                                                       p_fin.shape[ 0 ] ),
                                mpi_info )
        
    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_A20_B20): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       Q.shape[ 0 ] ),
                                mpi_info )

    # kineFactor[ b, p, Q, ratio, [A20,B20] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):

            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                p_ini = p - q
                    
                qSq = np.dot( q, q )

                pSq_ini = np.dot( p_ini, p_ini )
                pSq_fin = np.dot( p, p )

                # CJL:HERE

                kineFactor[ b, ip, iq ] \
                    = [ [ 1./4. * ( mEff[ b ] ** 2
                                    - 2 * ( energy( mEff[ b ],
                                                    pSq_fin, L )
                                            + energy( mEff[ b ],
                                                      pSq_ini, L ) ) ** 2
                                    + energy( mEff[ b ],
                                              pSq_fin, L )
                                    * energy( mEff[ b ],
                                              pSq_ini, L )
                                    - p[ 0 ] * p_ini[ 0 ]
                                    - p[ 1 ] * p_ini[ 1 ]
                                    - p[ 2 ] * p_ini[ 2 ] ),
                          mEff[ b ] ** 2
                          - 2 * ( energy( mEff[ b ],
                                          pSq_fin, L )
                                  - energy( mEff[ b ],
                                            pSq_ini, L ) ) ** 2
                          - energy( mEff[ b ],
                                    pSq_fin, L )
                          * energy( mEff[ b ],
                                    pSq_ini, L )
                          + p[ 0 ] * p_ini[ 0 ]
                          + p[ 1 ] * p_ini[ 1 ]
                          + p[ 2 ] * p_ini[ 2 ] ],
                        [ 1./2. * ( energy( mEff[ b ],
                                            pSq_fin, L )
                                    + energy( mEff[ b ],
                                              pSq_ini, L ) )
                          * 2. * np.pi / L
                          * ( p[ 0 ] + p_ini[ 0 ] ),
                          2. * ( energy( mEff[ b ],
                                         pSq_fin, L )
                                 - energy( mEff[ b ],
                                           pSq_ini, L) )
                          * 2. * np.pi / L
                          * ( p[ 0 ] - p_ini[ 0 ] ) ],
                        [ 1./2. * ( energy( mEff[ b ],
                                            pSq_fin, L )
                                    + energy( mEff[ b ],
                                              pSq_ini, L ) )
                          * 2. * np.pi / L
                          * ( p[ 1 ] + p_ini[ 1 ] ),
                          2. * ( energy( mEff[ b ],
                                         pSq_fin, L )
                                 - energy( mEff[ b ],
                                           pSq_ini, L) )
                          * 2. * np.pi / L
                          * ( p[ 1 ] - p_ini[ 1 ] ) ],
                        [ 1./2. * ( energy( mEff[ b ],
                                            pSq_fin, L )
                                    + energy( mEff[ b ],
                                              pSq_ini, L ) )
                          * 2. * np.pi / L
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( energy( mEff[ b ],
                                         pSq_fin, L )
                                 - energy( mEff[ b ],
                                           pSq_ini, L) )
                          * 2. * np.pi / L
                          * ( p[ 2 ] - p_ini[ 2 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 1 ] + p_ini[ 1 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] - p_ini[ 0 ] )
                          * ( p[ 1 ] - p_ini[ 1 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] + p_ini[ 0 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 0 ] - p_ini[ 0 ] )
                          * ( p[ 2 ] - p_ini[ 2 ] ) ],
                        [ 1./2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] + p_ini[ 1 ] )
                          * ( p[ 2 ] + p_ini[ 2 ] ),
                          2. * ( 2. * np.pi / L ) ** 2
                          * ( p[ 1 ] - p_ini[ 1 ] )
                          * ( p[ 2 ] - p_ini[ 2 ] ) ] ] \
                    / np.repeat( ratio_err[ ip, iq ] ** 2,
                                 2).reshape( ratioNum, 2 ) \
                    / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


def calcFormFactors_SVD( kineFactor_loc, ratio, ratio_err, Qsq_where,
                         formFactor, ratioSign, mpi_info ):

    # kineFactor_loc[ b_loc, p, q, ratio, [ F1, F2 ] ]
    # ratio[ b, p, q, ratio ]
    # ratio_err[ p, q, ratio ]
    # mpi_info
    
    comm = mpi_info[ 'comm' ]

    binNum = mpi_info[ 'binNum_glob' ]
    binNum_loc = mpi_info[ 'binNum_loc' ]

    binList_loc = mpi_info[ 'binList_loc' ]

    recvCount = mpi_info[ 'recvCount' ]
    recvOffset = mpi_info[ 'recvOffset' ]

    qNum = kineFactor_loc.shape[ 2 ]

    QsqNum = len( Qsq_where )
    ratioNum = kineFactor_loc.shape[ -2 ]

    ratio_loc = ratio[ binList_loc ]

    # kineFactor[ b, p, q, r, [ F1, F2 ] ]

    kineFactor = np.zeros( ( binNum, ) + kineFactor_loc.shape[ 1: ] )
    
    comm.Allgatherv( kineFactor_loc,
                     [ kineFactor,
                       recvCount \
                       * np.prod( kineFactor_loc.shape[ 1: ] ),
                       recvOffset \
                       * np.prod( kineFactor_loc.shape[ 1: ] ),
                       MPI.DOUBLE ] )

    # Repeat error for each bin

    ratio_err_loc = np.array( [ ratio_err ] * binNum_loc )
    ratio_err_loc = ratio_err_loc.reshape( ( binNum_loc, )
                                           + ratio.shape[ 1: ] )

    ratio_err_glob = np.array( [ ratio_err ] * binNum )
    ratio_err_glob = ratio_err_glob.reshape( ratio.shape )

    # F_loc[ b_loc, qs, [ F1, F2 ] ]

    F_loc = np.zeros( ( binNum_loc, QsqNum, 2 ), dtype=float )
    #F_loc = np.zeros( ( binNum_loc, QsqNum, 4, 2 ), dtype=float )
    
    Qsq_good = np.full( QsqNum, False, dtype=bool )
    #Qsq_good = np.full( ( QsqNum, 4 ), False, dtype=bool )

    curr_str = [ "g0", "gx", "gy", "gz" ]

    if True:
    #for ic in range( 4 ):
        """
        #mpi_fncs.mpiPrint(curr_str[ic],mpi_info)

        # Calculate F1 and F2 for Q^2=0 (only needed for testing
        # when there is only one element for Q^2=0)
    
        sum_axes = tuple( range( 1,
                                 ratio_loc[ :,
                                            Qsq_where[ 0 ],
                                            ic ].ndim ) )
        
        for f in range( 2 ):

            F_loc[ :, 0, ic, f ] \
                = np.average( ratio_loc[ :,
                                         Qsq_where[ 0 ],
                                         ic ]
                              / ratio_err_loc[ :,
                                               Qsq_where[ 0 ],
                                               ic ] ** 2
                              / kineFactor_loc[ :,
                                                Qsq_where[ 0 ],
                                                ic, f ],
                              axis=sum_axes )
        
        Qsq_good[ 0, ic ] = True
        """
        for iqs in range( QsqNum ):
        #for iqs in range( 1, QsqNum ):
         
            # kineFactor_Qsq[ b, Q^2[ qs ], r, [ F1, F2 ] ]

            kineFactor_Qsq \
                = kineFactor[ :, Qsq_where[ iqs ], :, : ]

            # ratio_Qsq[ b, Q^2[ qs ], r ]

            ratio_Qsq = ratio[ :, Qsq_where[ iqs ], : ]
            ratio_err_Qsq = ratio_err_glob[ :, Qsq_where[ iqs ], : ] 

            # Number of combinations of p and q
            # for this value of Q^2

            QsqNum_Qsq = kineFactor_Qsq.shape[ 1 ]

            """
            # kineFactor_Qsq[ b, Q^2[ qs ], [ F1, F2 ] ]
        
            kineFactor_Qsq = kineFactor[ :, Qsq_where[ iqs ], ic, : ]

            # ratio_Qsq[ b, Q^2[ qs ] ]

            ratio_Qsq = ratio[ :, Qsq_where[ iqs ], ic ]
            ratio_err_Qsq = ratio_err_glob[ :, Qsq_where[ iqs ], ic ] 

            # Number of combinations of p and q
            # for this value of Q^2

            QsqNum_Qsq = kineFactor_Qsq.shape[ 1 ]
        
            #mpi_fncs.mpiPrint(kineFactor_Qsq.shape,mpi_info)
            #mpi_fncs.mpiPrint(ratio_Qsq.shape,mpi_info)
            #mpi_fncs.mpiPrint(ratio_err_Qsq.shape,mpi_info)

            """
            # kineFactor_Qsq[ b, Q^2[ qs ], r, [ F1, F2 ] ]
            # -> kineFactor_Qsq[ b, Q^2[ qs ] * r, [ F1, F2 ] ]

            kineFactor_Qsq = kineFactor_Qsq.reshape( binNum,
                                                     QsqNum_Qsq * ratioNum,
                                                     2 )

            # ratio_Qsq[ b, Q^2[ qs ], r ]
            # -> ratio_Qsq[ b, Q^2[ qs ] * r ]

            ratio_Qsq = ratio_Qsq.reshape( binNum, QsqNum_Qsq * ratioNum )
            ratio_err_Qsq \
                = ratio_err_Qsq.reshape( binNum, QsqNum_Qsq * ratioNum )

            where_good = np.full( ( QsqNum_Qsq * ratioNum ), False,
                                  dtype=bool )
            #where_good = np.full( ( QsqNum_Qsq ), False, dtype=bool )

            ratioSign_arr \
            = np.array( ratio_Qsq.size
                        * [ ratioSign ] ).reshape( ratio_Qsq.shape )

            #mpi_fncs.mpiPrint(kineFactor_Qsq[0],mpi_info)
            #mpi_fncs.mpiPrint(ratio_Qsq[0],mpi_info)
            #mpi_fncs.mpiPrint(ratio_err_Qsq[0],mpi_info)

            # Loop over Q^2 and ratio
            for iqr in range( QsqNum_Qsq * ratioNum ):
            #for iqr in range( QsqNum_Qsq ):

                #CJL:HERE

                if formFactor == "GE_GM":

                    # Check that all bins meet requirements:
                    # K * R has right sign
                    # error/ratio is < 0.3
                    # 0.25 < |R| < 1.5
                    
                    #where_good[ iqr ] \
                    #    = np.all( ( np.sign( kineFactor_Qsq[ :, iqr, 0 ]
                    #                         * ratio_Qsq[ :, iqr ] )
                    #                == np.sign( ratioSign_arr[ :, iqr ] ) )
                    #              & ( ratio_err_Qsq[ :, iqr ]
                    #                  / np.abs( ratio_Qsq[ :, iqr ] )
                    #                  < 0.2 )
                    #              & ( np.abs( ratio_Qsq[ :, iqr ]
                    #                          / ratio_err_Qsq[ :, iqr ] ** 2
                    #                          / kineFactor_Qsq[ :, iqr, 0 ] )
                    #                  < 1.5 )
                    #              & ( np.abs( ratio_Qsq[ :, iqr ]
                    #                          / ratio_err_Qsq[ :, iqr ] ** 2
                    #                          / kineFactor_Qsq[ :, iqr, 0 ] )
                    #                  > 0.25 ) )
                    where_good[ iqr ] \
                        = np.all( ratio_err_Qsq[ :, iqr ]
                                  / np.abs( ratio_Qsq[ :, iqr ] )
                                  < 0.2 )

                elif formFactor == "A20_B20":

                    where_good[ iqr ] \
                        = np.all( ratio_err_Qsq[ :, iqr ]
                                  / np.abs( ratio_Qsq[ :, iqr ] )
                                  < 0.3 )
                    #where_good[ iqr ] = True
                    
            # End loop over Q^2 and ratio

            #mpi_fncs.mpiPrint(where_good,mpi_info)

            # Skip this Q^2 if there are no good elements

            if not np.any( where_good ):

                continue

            #mpi_fncs.mpiPrint(ratio_Qsq[0]/
            #                  kineFactor_Qsq[0,:,0]/
            #                  ratio_err_Qsq[0]**2,
            #                  mpi_info)

            # Change to local
                
            kineFactor_Qsq = kineFactor_loc[ :, Qsq_where[ iqs ], :, : ]

            ratio_Qsq = ratio_loc[ :, Qsq_where[ iqs ], : ]
            ratio_err_Qsq = ratio_err_loc[ :, Qsq_where[ iqs ], : ]
            
            # kineFactor_Qsq[ b_loc, Q^2[ qs ], r, [ F1, F2 ] ]
            # -> kineFactor_Qsq[ b_loc, Q^2[ qs ] * r, [ F1, F2 ] ]

            kineFactor_Qsq = kineFactor_Qsq.reshape( binNum_loc,
                                                     QsqNum_Qsq * ratioNum,
                                                     2 )

            # ratio_Qsq[ b, Q^2[ qs ], r ]
            # -> ratio_Qsq[ b, Q^2[ qs ] * r ]

            ratio_Qsq = ratio_Qsq.reshape( binNum_loc,
                                           QsqNum_Qsq * ratioNum )
            ratio_err_Qsq \
                = ratio_err_Qsq.reshape( binNum_loc, QsqNum_Qsq * ratioNum )

            # Select good elements

            kineFactor_Qsq = kineFactor_Qsq[ :, where_good, : ]
            ratio_Qsq = ratio_Qsq[ :, where_good ]
            ratio_err_Qsq = ratio_err_Qsq[ :, where_good ]
            """
            # Change to local
                
            kineFactor_Qsq = kineFactor_loc[ :, Qsq_where[ iqs ], ic, : ]

            ratio_Qsq = ratio_loc[ :, Qsq_where[ iqs ], ic ]
            ratio_err_Qsq = ratio_err_loc[ :, Qsq_where[ iqs ], ic ]
            
            # Select good elements

            kineFactor_Qsq = kineFactor_Qsq[ :, where_good, : ]
            ratio_Qsq = ratio_Qsq[ :, where_good ]
            ratio_err_Qsq = ratio_err_Qsq[ :, where_good ]

            #mpi_fncs.mpiPrint(kineFactor_Qsq.shape,mpi_info)
            #mpi_fncs.mpiPrint(ratio_Qsq.shape,mpi_info)
            #mpi_fncs.mpiPrint(ratio_err_Qsq.shape,mpi_info)
            """
            #mpi_fncs.mpiPrint(iqs,mpi_info)
            #mpi_fncs.mpiPrint(kineFactor_Qsq[0],mpi_info)

            # Perform SVD

            u, s, vT = np.linalg.svd( kineFactor_Qsq, full_matrices=False )

            #mpi_fncs.mpiPrint(u.shape,mpi_info)
            #mpi_fncs.mpiPrint(u[0],mpi_info)
            #mpi_fncs.mpiPrint(s.shape,mpi_info)
            #mpi_fncs.mpiPrint(s[0],mpi_info)
            #mpi_fncs.mpiPrint(vT.shape,mpi_info)
            #mpi_fncs.mpiPrint(vT[0],mpi_info)

            # Calculate ( v s^-1 u^T )^T
                
            uT = np.transpose( u, ( 0, 2, 1 ) )
            v = np.transpose( vT, ( 0, 2, 1 ) )
            
            #mpi_fncs.mpiPrint(uT.shape,mpi_info)
            #mpi_fncs.mpiPrint(uT[0],mpi_info)
            
            smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
            smat_inv = np.zeros( ( binNum_loc, )
                                 + np.transpose( smat ).shape )
    
            for b in range( binNum_loc ):
                    
                smat[ :vT.shape[ -2 ], :vT.shape[ -2 ] ] = np.diag( s[ b ] )
                    
                smat_inv[ b ] = np.linalg.pinv( smat )
    
            # End loop over bins

            # decomp[ b_loc, Q^2[qs]*ratio, [ F1, F2 ] ]
                    
            #mpi_fncs.mpiPrint(v.shape,mpi_info)
            #mpi_fncs.mpiPrint(v[0],mpi_info)
            #mpi_fncs.mpiPrint(smat_inv.shape,mpi_info)
            #mpi_fncs.mpiPrint(smat_inv[0],mpi_info)
            #mpi_fncs.mpiPrint(uT.shape,mpi_info)
            #mpi_fncs.mpiPrint(uT[0],mpi_info)
            
            decomp = np.transpose( v @ smat_inv @ uT, ( 0, 2, 1 ) )

            #mpi_fncs.mpiPrint(decomp.shape,mpi_info)
            #mpi_fncs.mpiPrint(decomp[0],mpi_info)

            sum_axes = tuple( range( 1, ratio_Qsq.ndim ) )
 
            #mpi_fncs.mpiPrint( np.average( decomp[ ..., 0 ]
            #                               * ratio_Qsq
            #                               / ratio_err_Qsq ** 2,
            #                               axis=0 ),
            #                   mpi_info )
            
            for iff in range( 2 ):

                #CJL:HERE
                #mpi_fncs.mpiPrint( "kineFactor",mpi_info )
                #mpi_fncs.mpiPrint( kineFactor_Qsq[ 0, ..., iff ].reshape(ratio_Qsq[0].size//7,7),
                #                   mpi_info )
                #mpi_fncs.mpiPrint( "decomp",mpi_info )
                #mpi_fncs.mpiPrint( decomp[ 0, ..., iff ].reshape(ratio_Qsq[0].size//7,7),
                #                   mpi_info )
                #mpi_fncs.mpiPrint( "ratio",mpi_info )
                #mpi_fncs.mpiPrint( ratio_Qsq[0].reshape(ratio_Qsq[0].size//7,7),
                #                   mpi_info )
                #mpi_fncs.mpiPrint( "decomp*ratio",mpi_info )
                #mpi_fncs.mpiPrint( np.array( decomp[ 0, ..., iff ]
                #                             * ratio_Qsq[0]
                #                             / ratio_err_Qsq[0] ** 2 ).reshape(ratio_Qsq[0].size//7,7),
                #                   mpi_info )
                #mpi_fncs.mpiPrint( decomp[ 0, ..., iff ]
                #                            * ratio_Qsq[0]
                #                            / ratio_err_Qsq[0] ** 2,
                #                   mpi_info )
                #mpi_fncs.mpiPrint( np.sum( decomp[ 0, ..., iff ]
                #                           * ratio_Qsq[0] 
                #                           / ratio_err_Qsq[0] ** 2 ),
                #                   mpi_info )
                
                F_loc[ :, iqs, iff ] = np.sum( decomp[ ..., iff ]
                                               * ratio_Qsq
                                               / ratio_err_Qsq ** 2,
                                               axis=sum_axes )
                #F_loc[ :, iqs, ic, iff ] = np.sum( decomp[ ..., iff ]
                #                                   * ratio_Qsq,
                #                                   axis=sum_axes )
                #/ ratio_err_Qsq ** 2

                #F_loc[ :, iqs, ic, iff ] \
                #    = np.average( ratio_Qsq
                #                  / kineFactor_Qsq[ ..., 0 ],
                #                  axis=sum_axes )

            Qsq_good[ iqs ] = True
            #Qsq_good[ iqs, ic ] = True

            #if iqs == 0:

            #    decomp_loc = decomp

            #else:

            #    decomp_loc = np.concatenate( ( decomp_loc, decomp ), axis=1 )

        # End loop over Q^2
    # End loop over current

    #decomp_loc = np.asarray( decomp_loc.reshape( binNum_loc,
    #                                             qNum,
    #                                             ratioNum, 2 ),
    #                         order='c' )

    return F_loc, Qsq_good


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


# avgXKineFactor = 2E/(1/2*m^2-2*E^2)

def avgXKineFactor( mEff, momSq, L ):

    return 2.0 * energy( mEff, momSq, L ) \
        / ( 0.5 * mEff ** 2 \
            - 2.0 * energy( mEff, momSq, L ) ** 2 )


def avgX2KineFactor( mEff, momSq, L ):

    return -energy( mEff, momSq, L ) / mEff


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


def calcAvgX2( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX2 = np.zeros( threep.shape )

    preFactor = avgX2KineFactor( mEff, momSq, L )

    for t in range( threep.shape[ 1 ] ):
           
        avgX2[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX2


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

    preFactor = avgXKineFactor( mEff, momSq, L )

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


def calcMatrixElemEM_ratio( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    ratio = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        ratio[ :, t ] = threep[ :, t ] / twop_tsink

    return ratio


def calcMatrixElemEM_twopFit( threep, tsink, c0, E0 ):
#def calcMatrixElemEM_twopFit( threep, tsink, c0, c1, E0, E1 ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    #c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    #E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    ratio = threep / twopFit( c0_cp, E0_cp, tsink )
    #ratio = threep / twopFit( c0_cp, c1_cp, E0_cp, E1_cp, tsink )

    return ratio


def calcMellin_twopFit( threep, tsink, mEff, momSq, L, \
                        c0, E0, moment ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( threep.shape )
    E0_cp = np.repeat( E0, T ).reshape( threep.shape )

    if moment == 1 or moment == "avgX":

        preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                               T ).reshape( threep.shape )

    elif moment == 2 or moment == "avgX2":

        preFactor = -1.0

    elif moment == 3 or moment == "avgX3":

        preFactor = -1.0        

    return preFactor * threep \
        / twopFit( c0_cp, E0_cp, tsink )


def calcAvgX_twopFit( threep, tsink, mEff, momSq, L, \
                      c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                           T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX


def calcAvgX2_twopFit( threep, tsink, mEff, momSq, L, \
                       c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    #preFactor = np.repeat( avgX2KineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    preFactor = -1.0

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX2 = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX2


def calcAvgX_twopTwoStateFit( threep, tsink, mEff, momSq, L, T, \
                              c0, c1, E0, E1 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # c0[ b ]
    # c1[ b ]
    # E0[ b ]
    # E1[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor=1.0
    #preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / fit.twoStateTwop( tsink, T, \
                               c0_cp, c1_cp, \
                               E0_cp, E1_cp )
    return avgX


def calcAvgX_twopOneStateFit( threep, tsink, mEff, momSq, L, T, G, E ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # G[ b ]
    # E[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor = energy( mEff, momSq, L ) \
                / ( mEff * ( 0.5 * mEff ** 2 \
                             - 2.0 * energy( mEff, momSq, L ) ** 2 ) )

    G_cp = np.repeat( G, T ).reshape( binNum, T )
    E_cp = np.repeat( E, T ).reshape( binNum, T )
    
    avgX = preFactor * threep \
           / fit.oneStateTwop( tsink, T, \
                               G_cp, E_cp )

    return avgX


def calcAvgX_twoStateFit( a00, c0, mEff, momSq, L, ZvD1 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD1 * avgXKineFactor( mEff, momSq, L ) \
        * a00 / c0

def calcAvgX2_twoStateFit( a00, c0, mEff, momSq, L, ZvD2 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD2 * avgX2KineFactor( mEff, momSq, L ) \
        * a00 / c0

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


def calcFormFactorRatio( threep, twop, tsink ):
    
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


def calcFormFactorRatio_twopFit( threep, c0, mEff, tsink, p_fin,
                         Q, pSq_twop, L, mpi_info ):
    
    # threep[ ..., p, Q, r, t ]
    # c0[ ..., p^2 ]
    # mEff[ ... ]
    # tsink
    # p_fin[ p, pi ]
    # Q[ q, qi ]
    # pSq_twop[ ps ]
    # L

    QNum = threep.shape[ -3 ]
    ratioNum = threep.shape[ -2 ]
    
    if QNum != len( Q ):

        error_template = "Error (physQuants.calcFormFactorRatio_twopFit): " \
                         + "length of threep Q dimension {} " \
                         + "does not match length of " \
                         + "Q {}."

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       len( Q ) ),
                                mpi_info )

    ratio = np.zeros( threep.shape[ :-1 ] + ( tsink + 1, ) )

    # Calculate twop from fit parameters
    # twop[ ..., p^2, r, t ]

    twop = np.zeros( c0.shape + ( ratioNum, tsink + 1 ) )

    # Loop over twop p^2
    for ps, ips in fncs.zipXandIndex( pSq_twop ):
        # Loop over time
        for t in range( tsink + 1 ):

            twop_tmp = twopFit( c0[ ..., ips ],
                                energy( mEff, ps, L ),
                                t )
            twop_tmp = np.repeat( twop_tmp, ratioNum )

            twop[ ..., ips, :, t ] \
                = twop_tmp.reshape( twop[ ..., ips,
                                          :, t ].shape )

        # End loop over t
    # End loop over twop p^2

    for p, ip in fncs.zipXandIndex( p_fin ):
        
        pSq_fin = np.dot( p, p )

        pSq_fin_where = np.where( pSq_twop == pSq_fin )

        for q, iq, in fncs.zipXandIndex( Q ):

            pSq_ini = np.dot( p - q, p - q )

            pSq_ini_where = np.where( pSq_twop == pSq_ini )

            for t in range( tsink + 1 ):

                ratio[..., ip, iq, :, t] \
                    = threep[ ..., ip, iq, :, t ] \
                    / twop[ ..., pSq_fin_where, :, tsink ] \
                    * np.sqrt( twop[ ..., pSq_ini_where, :, tsink - t ]
                               * twop[ ..., pSq_fin_where, :, t ]
                               * twop[ ..., pSq_fin_where, :, tsink ]
                               / ( twop[ ..., pSq_fin_where, :, tsink - t ]
                                   * twop[ ..., pSq_ini_where, :, t ]
                                   * twop[ ..., pSq_ini_where, :, tsink ] ) )
            
            # End loop over t
        # End loop ovet Q
    # End loop over p

    return ratio


# Calculate the electromagnetic form factor.

# threep:

def calcEMFF( threep, twop, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        #factor = 1.0
        factor = 2 * energy / ( energy + mEff )
        #factor = 4.0 * np.sqrt( energy * mEff ) / ( energy + mEff )
        
        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / twop[ 0, :, tsink ] \
                             * np.sqrt( twop[ q, :, tsink - t ] * twop[ 0, :, t ] * twop[ 0, :, tsink ] \
                                        / ( twop[ 0, :, tsink - t ] * twop[ q, :, t ] * twop[ q, :, tsink ] ) )

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

