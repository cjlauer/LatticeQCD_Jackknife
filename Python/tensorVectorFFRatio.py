import sys
import time
import numpy as np
import argparse as argp
import functions as fncs
import readWrite as rw
import lqcdjk_fitting as fit

np.set_printoptions(threshold=sys.maxsize)


#########################
# Parse input arguments #
#########################


parser = argp.ArgumentParser( description="Calculate the ratio "
                              + "of the tensor and vector form factors." )

parser.add_argument( "data_dir", action='store', type=str )

parser.add_argument( "formFactor_template", action='store',
                     help="Comma seperated list of form factor filenames "
                     + "for each momentum frame",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "monopole_params_template", action='store',
                     help="Filename for monopole fit parameters for "
                     + "tensor form factor",
                     type=str )

parser.add_argument( "particle", action='store',
                     help="Particle to calculate",
                     type=str )

parser.add_argument( "flavor", action='store',
                     help="Flavor to calculate",
                     type=str )

parser.add_argument( "Qsq_last", action='store',
                     type=lambda s: [float(item) for item in s.split(',')],
                     help="Comma seperated list of last Q^2 to include "
                     + "in the ratio calculation for each momentum frame." )

parser.add_argument( "bin_num", action='store',
                     help="Number of bins in input files",
                     type=int )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat",
                     help="Template for output files. '*' will be "
                     + "replaced with text depending on output data." )

args = parser.parse_args()


#########
# Setup #
#########


# Input directories and filename templates

data_dir = args.data_dir

F_template = args.formFactor_template

monopole_params_template = args.monopole_params_template

particle = args.particle

flavor = args.flavor

pSq = [ 0, 3 ]
pSqNum = len( pSq )

if len( F_template ) != pSqNum:

    error_template = "Error (vectorTensorFFRatio.py): length of F_template {} " \
                     "does not match length of form factor filename templates {}."

    print( error_template.format( len( F_template ), pSqNum ) )

    exit()

if len( args.Qsq_last ) != pSqNum:

    error_template = "Error (vectorTensorFFRatio.py): length of args.Qsq_last {} " \
                     "does not match length of form factor filename templates {}."

    print( error_template.format( len( args.Qsq_last ), pSqNum ) )

    exit()

binNum = args.bin_num

output_template = args.output_template

formFactors = [ "tensorFF", "GE", "GE" ]

particle_flavor = [ "{}_{}".format( particle, flavor ),
                    particle,
                    "{}_{}".format( particle, flavor ) ]

###################
# Read data files #
###################


# F[ ff ][ ps ][ b, Qsq ]

F = [ [ [] for ps in F_template ]
      for ff in formFactors ]

# Qsq[ ff ][ ps ][ b, Qsq ]

Qsq = [ [ [] for ps in F_template ]
            for ff in formFactors ]

# M[ ff, b ]

M = [ [] for ff in formFactors ]

# F0[ ff, b ]

F0 = [ [] for ff in formFactors ]

# Loop over form factors
for ff, iff in fncs.zipXandIndex( formFactors ):
    # Loop over p^2 filename templates
    for template, ips in fncs.zipXandIndex( F_template ):

        F_filename = template.format( data_dir, ff, particle_flavor[ iff ] )

        Qsq[ iff ][ ips ], F[ iff ][ ips ] \
            = rw.readFormFactorFile_ASCII( F_filename, binNum )

    # End loop over p^2 filename templates

    # Tensor monopole fit parameters
    # M[ b ], F0[ b ]

    monopole_params_filename = monopole_params_template.format( data_dir, ff,
                                                            particle_flavor[ iff ] )

    M[ iff ] = rw.readNthDataCol( monopole_params_filename, 0 )
    F0[ iff ] = rw.readNthDataCol( monopole_params_filename, 1 )

# End loop over form factors


##################################
# Sort form factors based on Q^2 #
##################################


# Loop over form factors
for ff, iff in fncs.zipXandIndex( formFactors ):

    # Loop over p^2
    for ips in range( pSqNum ):

        # Sort by Qsq

        # Loop over bins
        for b in range( binNum ):

            iq_sort = np.argsort( Qsq[ iff ][ ips ][ b ] )
            
            Qsq[ iff ][ ips ][ b ] = Qsq[ iff ][ ips ][ b, iq_sort ]
            
            F[ iff ][ ips ][ b ] = F[ iff ][ ips ][ b, iq_sort ]

        # End loop over bins

        # Average over bins
        # F_avg[ Qsq ]

        F_avg = np.average( F[ iff ][ ips ], axis=0 )

        # Remove elements where form factors are zero
        
        F[ iff ][ ips ] = F[ iff ][ ips ][ :, F_avg != 0. ]
        Qsq[ iff ][ ips ] = Qsq[ iff ][ ips ][ :, F_avg != 0. ]

    # End loop over p^2
# End loop over form factors


####################################################
# Get Qsq shared by tensor and vector form factors #
####################################################


Qsq_avg = [ [] for p in pSq ]
ratio_avg = [ [] for p in pSq ]
ratio_err = [ [] for p in pSq ]

Qsq_flavor_avg = [ [] for p in pSq ]
ratio_flavor_avg = [ [] for p in pSq ]
ratio_flavor_err = [ [] for p in pSq ]

# Loop over p^2
for ps, ips in fncs.zipXandIndex( pSq ):

    Qsq_psq = [ [] for b in range( binNum ) ]
    F_tensor = [ [] for b in range( binNum ) ]
    F_vector = [ [] for b in range( binNum ) ]

    Qsq_flavor_psq = [ [] for b in range( binNum ) ]
    F_tensor_flavor = [ [] for b in range( binNum ) ]
    F_vector_flavor = [ [] for b in range( binNum ) ]

    # Loop over bins
    for b in range( binNum ):

        # Q^2 on this bin

        Qsq_tensor = Qsq[ 0 ][ ips ][ b ]
        Qsq_vector = Qsq[ 1 ][ ips ][ b ]
        Qsq_vector_flavor = Qsq[ 2 ][ ips ][ b ]

        # Qsq where are shared

        Qsq_psq[ b ] \
            = np.intersect1d( Qsq_tensor, Qsq_vector )
        Qsq_flavor_psq[ b ] \
            = np.intersect1d( Qsq_tensor, Qsq_vector_flavor )

        # Find boolean array of where shared Q^2 are

        where_shared_tensor = np.full( len( Qsq_tensor ), False, dtype=bool )
        where_shared_vector = np.full( len( Qsq_vector ), False, dtype=bool )

        where_shared_tensor_flavor = np.full( len( Qsq_tensor ), False, dtype=bool )
        where_shared_vector_flavor = np.full( len( Qsq_vector_flavor ), False, dtype=bool )

        for qs in Qsq_psq[ b ]:

            where_shared_tensor = np.logical_or( where_shared_tensor,
                                                 Qsq_tensor == qs )
            where_shared_vector = np.logical_or( where_shared_vector,
                                                 Qsq_vector == qs )

        for qs in Qsq_flavor_psq[ b ]:

            where_shared_tensor_flavor = np.logical_or( where_shared_tensor_flavor,
                                                 Qsq_tensor == qs )
            where_shared_vector_flavor = np.logical_or( where_shared_vector_flavor,
                                                 Qsq_vector_flavor == qs )

        # Form factors on this bin

        F_tensor_b = F[ 0 ][ ips ][ b ]
        F_vector_b = F[ 1 ][ ips ][ b ]
        F_vector_flavor_b = F[ 2 ][ ips ][ b ]

        # Form factors on this bin which share Q^2

        F_tensor[ b ] = F_tensor_b[ where_shared_tensor ]
        F_vector[ b ] = F_vector_b[ where_shared_vector ]

        F_tensor_flavor[ b ] = F_tensor_b[ where_shared_tensor_flavor ]
        F_vector_flavor[ b ] = F_vector_flavor_b[ where_shared_vector_flavor ]

    # End loop over bins

    # Qsq_psq[ b, Qsq ]

    Qsq_psq = np.array( Qsq_psq )

    # Qsq_flavor_psq[ b, Qsq_flavor ]

    Qsq_flavor_psq = np.array( Qsq_flavor_psq )

    # F_tensor[ b, Qsq ]

    F_tensor = np.array( F_tensor )

    # F_vector[ b, Qsq ]

    F_vector = np.array( F_vector )

    # F_tensor_flavor[ b, Qsq ]

    F_tensor_flavor = np.array( F_tensor_flavor )

    # F_vector_flavor[ b, Qsq ]

    F_vector_flavor = np.array( F_vector_flavor )

    # Average over bins

    Qsq_avg[ ips ] = np.average( Qsq_psq, axis=0 )
    Qsq_flavor_avg[ ips ] = np.average( Qsq_flavor_psq, axis=0 )


    ####################
    # Calculate ratios #
    ####################


    ratio = F_tensor / F_vector

    ratio_flavor = F_tensor_flavor / F_vector_flavor

    # Average over bins

    ratio_avg[ ips ] = np.average( ratio, axis=0 )
    ratio_err[ ips ] = fncs.calcError( ratio, binNum )

    ratio_flavor_avg[ ips ] = np.average( ratio_flavor, axis=0 )
    ratio_flavor_err[ ips ] = fncs.calcError( ratio_flavor, binNum )

# End loop over p^2

# Cut ratio at args.Qsq_last

# Loop over p^2
for ps, ips in fncs.zipXandIndex( pSq ):

    Qsq_where = Qsq_avg[ ips ] <= args.Qsq_last[ ips ]

    Qsq_avg[ ips ] = Qsq_avg[ ips ][ Qsq_where ]

    ratio_avg[ ips ] = ratio_avg[ ips ][ Qsq_where ]
    ratio_err[ ips ] = ratio_err[ ips ][ Qsq_where ]

    Qsq_flavor_where = Qsq_flavor_avg[ ips ] <= args.Qsq_last[ ips ]

    Qsq_flavor_avg[ ips ] = Qsq_flavor_avg[ ips ][ Qsq_flavor_where ]

    ratio_flavor_avg[ ips ] = ratio_flavor_avg[ ips ][ Qsq_flavor_where ]
    ratio_flavor_err[ ips ] = ratio_flavor_err[ ips ][ Qsq_flavor_where ]

# End loop over p^2


#############################################
# Calculate monopole curves and their ratio #
#############################################
    

# Find highest Q^2 for all frames which will
# be the last Q^2 in curve

Qsq_last = 0.
Qsq_flavor_last = 0.

# Loop over p^2
for ps, ips in fncs.zipXandIndex( pSq ):

    Qsq_last = max( Qsq_avg[ ips ][ -1 ], Qsq_last )
    Qsq_flavor_last = max( Qsq_flavor_avg[ ips ][ -1 ], Qsq_flavor_last )

# End loop over p^2

curve_tensor, Qsq_curve = fit.calcMonopoleCurve( M[ 0 ], F0[ 0 ],
                                                 Qsq_last )

curve_vector, Qsq_curve = fit.calcMonopoleCurve( M[ 1 ], F0[ 1 ],
                                                 Qsq_last )

curve_vector_flavor, Qsq_flavor_curve \
    = fit.calcMonopoleCurve( M[ 2 ], F0[ 2 ], Qsq_flavor_last )

# Calculate curve ratio

curve_ratio = curve_tensor / curve_vector

curve_ratio_flavor = curve_tensor / curve_vector_flavor

# Average over bins

curve_ratio_avg = np.average( curve_ratio, axis=0 )
curve_ratio_err = fncs.calcError( curve_ratio, binNum )

curve_ratio_flavor_avg = np.average( curve_ratio_flavor, axis=0 )
curve_ratio_flavor_err = fncs.calcError( curve_ratio_flavor, binNum )


######################
# Write output files #
######################


# Loop over p^2
for ps, ips in fncs.zipXandIndex( pSq ):

    # Write ratio points

    output_filename = rw.makeFilename( output_template,
                                       "tensor_vector_ratio_{}_{}_psq{}",
                                       particle, flavor,
                                       ps)

    rw.writeAvgDataFile_wX( output_filename, Qsq_avg[ ips ],
                            ratio_avg[ ips ], ratio_err[ ips ] )

    output_filename = rw.makeFilename( output_template,
                                       "tensor_vector_ratio_{}_{}_{}_psq{}",
                                       particle, flavor, flavor,
                                       ps)

    rw.writeAvgDataFile_wX( output_filename, Qsq_flavor_avg[ ips ],
                            ratio_flavor_avg[ ips ], ratio_flavor_err[ ips ] )

# End loop over p^2

# Write ratio curve

output_filename = rw.makeFilename( output_template,
                                   "tensor_vector_curve_ratio_{}_{}",
                                   particle, flavor )

rw.writeAvgDataFile_wX( output_filename, Qsq_curve,
                        curve_ratio_avg, curve_ratio_err )

output_filename = rw.makeFilename( output_template,
                                   "tensor_vector_curve_ratio_{}_{}_{}",
                                   particle, flavor, flavor )

rw.writeAvgDataFile_wX( output_filename, Qsq_flavor_curve,
                        curve_ratio_flavor_avg, curve_ratio_flavor_err )

exit()
