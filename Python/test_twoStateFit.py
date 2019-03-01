import numpy as np
import argparse as argp
from scipy.optimize import leastsq
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 1.0

particle_list = [ "pion", "kaon" ]

format_list = [ "gpu", "cpu" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate quark momentum fraction <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "particle", action='store', help="Particle to calculate <x> for. Should be pion or kaon.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "bin_size", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_template = args.threep_template

twop_template = args.twop_template

particle = args.particle

tsink = args.t_sink

tsinkNum = len( tsink )

binSize = args.bin_size

output_template = args.output_template

dataFormat = args.data_format

# Check inputs

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Get configurations from given list or from given
# threep directory if list not given

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

binNum = configNum / binSize

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t ]

twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )[ :, 0, 0, ..., 0, 0 ]

print "Read two-point functions from HDF5 files"

# Jackknife
# twop_jk[ b, t ]

twop_jk = fncs.jackknife( twop, binSize )

twop_avg = np.average( twop_jk, axis=0 )

twop_err = np.std( twop_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

twop_ts = []

threep_jk = []

threep_avg = []

threep_err = []

for ts in tsink:
    
    #########################
    # Three-point functions #
    #########################

    # Get the real part of gxDx, gyDy, gzDz, and gtDt
    # three-point functions at zero-momentum
    # threep[ c, t ]

    if particle == "nucleon":

        if dataFormat == "cpu":

            filename_u_gxDx = threep_template + str( ts ) + ".up.h5"

            threep_u_gxDx = rw.getDatasets( threepDir, configList, filename_u_gxDx, \
                                            "=der:gxDx:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gyDy = threep_template + str( ts ) + ".up.h5"

            threep_u_gyDy = rw.getDatasets( threepDir, configList, filename_u_gyDy, \
                                            "=der:gyDy:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gzDz = threep_template + str( ts ) + ".up.h5"

            threep_u_gzDz = rw.getDatasets( threepDir, configList, filename_u_gzDz, \
                                            "=der:gzDz:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gtDt = threep_template + str( ts ) + ".up.h5"

            threep_u_gtDt = rw.getDatasets( threepDir, configList, filename_u_gtDt, \
                                            "=der:g0D0:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gxDx = threep_template + str( ts ) + ".dn.h5"

            threep_d_gxDx = rw.getDatasets( threepDir, configList, filename_d_gxDx, \
                                            "=der:gxDx:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gyDy = threep_template + str( ts ) + ".dn.h5"

            threep_d_gyDy = rw.getDatasets( threepDir, configList, filename_d_gyDy, \
                                            "=der:gyDy:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gzDz = threep_template + str( ts ) + ".dn.h5"

            threep_d_gzDz = rw.getDatasets( threepDir, configList, filename_d_gzDz, \
                                            "=der:gzDz:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gtDt = threep_template + str( ts ) + ".dn.h5"

            threep_d_gtDt = rw.getDatasets( threepDir, configList, filename_d_gtDt, \
                                            "=der:g0D0:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real
            
            threep_gxDx = threep_u_gxDx - threep_d_gxDx

            threep_gyDy = threep_u_gyDy - threep_d_gyDy

            threep_gzDz = threep_u_gzDz - threep_d_gzDz

            threep_gtDt = threep_u_gtDt - threep_d_gtDt

        else:

            print "GPU format not supported for nucleon, yet."

            exit()

    else: # Particle is meson

        if dataFormat == "gpu":

            threep_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_00", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

            threep_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_01", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
            threep_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_02", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

            threep_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                            "tsink_" + str( ts ), "oneD", "dir_03", \
                                            "up", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]

            threep_s_gxDx = np.array( [] )
            
            threep_s_gyDy = np.array( [] )
        
            threep_s_gzDz = np.array( [] )
    
            threep_s_gtDt = np.array( [] )

            if particle == "kaon":
            
                threep_s_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_00", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

                threep_s_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_01", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
                threep_s_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_02", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

                threep_s_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_03", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]
            
        elif dataFormat == "cpu":

            print "CPU format not supported for mesons, yet."
            
            exit()

    print "Read three-point functions from HDF5 files for tsink " + str( ts )

    # Subtract average over directions from gtDt

    threep = threep_gtDt - 0.25 * ( threep_gtDt + threep_gxDx + threep_gyDy + threep_gzDz )

    # Jackknife
    # threep_jk[ ts ][ b, t ]
    
    threep_jk.append( fncs.jackknife( threep, binSize ) )

    threep_avg = np.average( threep_jk[ -1 ], axis=0 )

    threep_err = np.std( threep_jk[ -1 ], axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    threep_outFilename = output_template.replace( "*", "threep_tsink" + str( ts ) )

    rw.writeAvgDataFile( threep_outFilename, threep_avg, threep_err )

# End loop over tsink

##################
# Two-state Fit  #
##################

# fitParams[ b, param ]
"""
threep_cp = []

for ts in range( tsinkNum ):

    threep_cp.append( threep_jk[ ts ][ :, 2:-2 ] )

fitParams = fncs.twoStateFit( twop_jk, threep_cp )
"""
fitParams = fncs.twoStateFit( twop_jk, threep_jk )

a00 = fitParams[ :, 0 ]
          
a01 = fitParams[ :, 1 ]

a11 = fitParams[ :, 2 ]
          
c0 = fitParams[ :, 3 ]

c1 = fitParams[ :, 4 ]
        
E0 = fitParams[ :, 5 ]
                
E1 = fitParams[ :, 6 ]

# Write curve with constant tsink

curve = np.zeros( ( binNum, tsinkNum, 50 ) )

t_i= np.zeros( ( tsinkNum, 50 ) )

for b in range( binNum ):

    for ts in range( tsinkNum ):

        t_i[ ts, : ] = np.linspace( -2, tsink[ ts ] + 2, 50 )

        for t in range( t_i.shape[ -1 ] ):

            curve[ b, ts, t ] = fncs.twoStateThreep( t_i[ ts, t ], tsink[ ts ], \
                                                     a00[ b ], a01[ b ], a11[ b ], \
                                                     E0[ b ], E1[ b ] )

# Average over bins

curve_avg = np.average( curve, axis=0 )

curve_err = np.std( curve, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
                        
fitParams_avg = np.average( fitParams, axis=0 )

fitParams_err = np.std( fitParams, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

#####################
# Write output file #
#####################

for ts in range( tsinkNum ):

    curveOutputFilename = output_template.replace( "*", "threep_twoStateFit_curve_tsink" + str( tsink[ ts ] ) )

    rw.writeAvgDataFile_wX( curveOutputFilename, t_i[ ts ], curve_avg[ ts ], curve_err[ ts ] )

avgXParamsOutputFilename = output_template.replace( "*", "threep_twoStateFitParams" )

rw.writeTSFParamsFile( avgXParamsOutputFilename, fitParams_avg, fitParams_err )
