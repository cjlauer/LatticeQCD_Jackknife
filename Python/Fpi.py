import math
import numpy as np
import argparse as argp
from os import listdir as ls
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

latticeSpacing = 0.093

latticeDim = 32

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Pion Electromagnetic Form Factor" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_template = args.threep_template

twop_template = args.twop_template

mEff_filename = args.mEff_filename

mEff_fitStart = args.mEff_fit_start

mEff_fitEnd = args.mEff_fit_end

tsink = args.t_sink

output_template = args.output_template

# Get configurations from given list or from given threep
# directory if list not given

configList = fncs.getConfigList( args.config_list, twopDir )

configNum = len( configList )

# Set timestep and bin number from effective mass file

T_fold, binNum = rw.detTimestepAndConfigNum( mEff_filename )

if configNum % binNum != 0:

    print( "Number of configurations " + str( configNum ) \
        + " not evenly divided by number of bins " + str( binNum ) \
        + " in effective mass file " + mEff_filename + ".\n" )

    exit()

binSize = configNum // binNum
"""
########################
# Fit effective masses #
########################

# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, binNum, T_fold )

# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( mEff_fitStart, mEff_fitEnd + 1 ), \
                                mEff[ b, mEff_fitStart : mEff_fitEnd + 1 ], \
                                0, w=mEff_err[ mEff_fitStart : mEff_fitEnd + 1 ] )

print( "Fit effective mass" )

#####################
# Average over bins #
#####################

# Fitted effective mass
# mEff_fit_avg
    
mEff_fit_avg = np.average( mEff_fit )

mEff_fit_err = np.std( mEff_fit ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )
"""
################
# Momenta list #
################

# Read momenta list from dataset
# Q[ c, Q ]

Q = rw.getDatasets( threepDir, configList, threep_template, "Momenta_list" )[ :, 0, 0, ... ]

QNum = Q.shape[1]

# Check that momenta agree across configurations

Qsq, Qsq_start, Qsq_end = fncs.processMomList( Q )

Q = Q[0]

#print( Qsq, Qsq_start, Qsq_end )

QsqNum = len( Qsq )

#print(QsqNum)

#print( Qsq )
#print( Qsq_start )
#print( Qsq_end )

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t, Q ]

twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )[ :, 0, 0, ..., 0 ]

print( "Read two-point functions from HDF5 files" )

T = twop.shape[ 1 ]

twop_jk = np.zeros( ( binNum, T, QNum ) )

for q in range( QNum ):
    """
    print("Q=({:+},{:+},{:+})".format(Q[q][0],Q[q][1],Q[q][2]))

    print(twop[5,:,q])

    """
    twop_jk[ ..., q ] = fncs.jackknife( twop[ ..., q ], binSize )

for q in range( Qsq_end[8]+1 ):

    avg = np.average( twop_jk[...,q], axis=0)
    err = fncs.calcError( twop_jk[...,q], binNum)
    
    twop_outFilename = output_template.replace( "*", "twop_Q_{:+}_{:+}_{:+}".format(Q[q][0],Q[q][1],Q[q][2]) )
    rw.writeAvgDataFile( twop_outFilename, avg, err )

# Average over equal Q^2
# twop_avg[ Q^2, b, t ]

twop_avg = fncs.averageOverQsq( twop_jk, Qsq_start, Qsq_end )
#twop_avg = fncs.averageOverQsq( twop, Qsq_start, Qsq_end )
"""
for q in range( QsqNum ):

    print("Q^2={}".format(Qsq[q]))

    print(twop_avg[q,5,:])
"""
# Jackknife
# twop_jk[ b, t, Q ]

#twop_jk = np.zeros( ( QsqNum, binNum, T ) )

#for q in range( QsqNum ):

#    twop_jk[ q, ... ] = fncs.jackknife( twop_avg[ q, ... ], binSize )
    
#print("AVG")

#for q in range( QsqNum ):

#    print(np.average(twop_jk[q,:,:],axis=0))

#print("ERR")

#for q in range( QsqNum ):

#    print(fncs.calcError(twop_jk[q,:,:],binNum))

for q in range( 20 ):

    avg = np.average( twop_avg[q], axis=0)
    err = fncs.calcError( twop_avg[q], binNum)
    
    twop_outFilename = output_template.replace( "*", "twop_Qsq{}".format(Qsq[q]) )
    rw.writeAvgDataFile( twop_outFilename, avg, err )

exit() # CJL:HERE

for ts in tsink:
    
    #########################
    # Three-point functions #
    #########################

    # Get the real part of gamma4 insertion three-point functions
    # threep[ c, t, Q ]

    threep = rw.getDatasets( threepDir, configList, \
                             threep_template, \
                             "tsink_" + str( ts ), \
                             "noether", \
                             "threep" )[ :, 0, 0, ..., 3, 0 ]

    #print(threep.shape)

    print( "Read three-point functions from HDF5 files for tsink " \
           + str( ts ) )


    # Average over equal Q^2
    # threep_avg[ Q^2, c, t ]
    
    #threep_avg = fncs.averageOverQsq( threep_jk, Qsq_start, Qsq_end )
    threep_avg = fncs.averageOverQsq( threep, Qsq_start, Qsq_end )

    # Jackknife
    # threep_jk[ b, t, Q ]
    
    threep_jk = np.zeros( ( QsqNum, binNum, ts+1 ) )

    for q in range( QsqNum ):

        threep_jk[ q, ... ] = fncs.jackknife( threep_avg[ q, ... ], \
                                              binSize )

    #threep_jk = np.zeros( ( binNum, ts+1, QNum ) )

    #for q in range( QNum ):

    #    threep_jk[ ..., q ] = fncs.jackknife( threep[ ..., q ], \
    #                                      binSize )

    #threep_jk_avg = np.average( threep_jk, axis=1 )
    #threep_jk_err = fncs.calcError( threep_jk, binNum, axis=1 )
    
    #print(threep_jk_avg.shape)

    #twop_jk_avg = np.average( twop_jk, axis=1 )
    #twop_jk_err = fncs.calcError( twop_jk, binNum, axis=1 )

    #########################
    # Calculate form factor #
    #########################

    #emff = pq.calcEMFF( threep_avg, twop_avg, Qsq, \
    #                    mEff_fit, ts, latticeDim )
    emff = pq.calcEMFF( threep_jk, twop_jk, Qsq, \
                        mEff_fit, ts, latticeDim )

    #####################
    # Average over bins #
    #####################

    # Electromagnetic form factor
    # em_avg[ Q^2, t ]

    emff_avg = np.average( emff, axis=1 )

    emff_err = fncs.calcError( emff, binNum, axis=1 )

    ######################
    # Write output files #
    ######################

    # Form factors for each Q^2
    
    emff_outFilename = output_template.replace( "*", "Fpi_tsink" + str( ts ) )

    rw.writeFormFactorFile( emff_outFilename, Qsq, emff )

    # Form factors for each Q^2 and bin

    emff_avg_outFilename = output_template.replace( "*", "avgFpi_tsink" + str( ts ) )
    rw.writeAvgFormFactorFile( emff_avg_outFilename, Qsq, emff_avg, emff_err )

    #threep_jk_avg_outFilename = output_template.replace( "*", "threep_jk_tsink" + str( ts ) )
    #rw.writeAvgFormFactorFile( threep_jk_avg_outFilename, Qsq, threep_jk_avg, threep_jk_err )

    # Fitted effective mass

    mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

    print( "Wrote output files for tsink " + str( ts ) )

# End loop over tsink
