import math
import numpy as np
import argparse as argp
from os import listdir as ls
from scipy.optimize import curve_fit
import functions as fncs
import physQuants as pq
import readWrite as rw

latticeSpacing = 0.093
latticeDim = 32

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Electromagnetic Form Factor" )

parser.add_argument( "Fpi_filename", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "fit_start", action='store', type=int )

parser.add_argument( "fit_end", action='store', type=int )

parser.add_argument( "-o", "--output", action='store', type=str, default="./Fpi_fit.dat" )

args = parser.parse_args()

Fpi_filename = args.Fpi_filename

fitStart = args.fit_start

fitEnd = args.fit_end

output = args.output

Qsq, binNum, timestepNum = rw.detQsqConfigNumAndTimestepNum( Fpi_filename )

QsqNum = Qsq.shape[ 0 ]

#############################################
# Fill matrices with values from data files #
#############################################

# Fpi[ q, b, t ]

Fpi = rw.readFormFactorFile( Fpi_filename, QsqNum, binNum, timestepNum )

# mEff

mEff = rw.readNthDataCol( args.mEff_filename, 0 )

print( "Read data files" )

# Fpi_err[ q, t ]

Fpi_err = np.std( Fpi, axis=1 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

########################
# Fit effective masses #
########################

Fpi_fit = np.zeros( ( QsqNum, binNum ) )

for q in range( QsqNum ):

    for b in range( binNum ):

        if True not in np.isnan( Fpi[ q, b, fitStart : fitEnd + 1 ] ) \
           and True not in np.isnan( Fpi_err[ q, fitStart : fitEnd + 1 ] ):

            Fpi_fit[ q, b ] = np.polyfit( range( fitStart, fitEnd + 1 ), \
                                          Fpi[ q, b, fitStart : fitEnd + 1 ], \
                                          0, w=Fpi_err[ q, fitStart : fitEnd + 1 ] )

        else:

            Fpi_fit[ q, b ] = np.nan


#####################
# Average over bins #
#####################

Fpi_fit_avg = np.average( Fpi_fit, axis=1 )

Fpi_fit_err = np.std( Fpi_fit, axis=1 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

######################
# Convert Q^2 to GeV #
######################

Qsq_GeV = pq.convertQsqToGeV( Qsq, latticeSpacing * mEff, latticeSpacing, latticeDim )

######################
# Write output files #
######################

# Fitted effective mass

rw.writeAvgDataFile_wX( output, Qsq_GeV, Fpi_fit_avg, Fpi_fit_err )

