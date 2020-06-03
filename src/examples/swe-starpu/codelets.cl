#define float_type float
#define real float
#define integer unsigned int

__kernel void updateUnknowns_opencl_kernel(
        __global float_type *mainH,
        __global float_type *mainHu,
        __global float_type *mainHv,
        __global float_type *netUpH,
        __global float_type *netUpHu,
        __global float_type *netUpHv,
        __global float *dt,
        unsigned int nX,
        unsigned int nY
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= nX || y >= nY) {
        return;
    }
    mainH[y * nX + x] -= *dt * netUpH[y * nX + x];
    mainHu[y * nX + x] -= *dt * netUpHu[y * nX + x];
    mainHv[y * nX + x] -= *dt * netUpHv[y * nX + x];

    if (mainH[y * nX + x] < 0.1) {
        mainH[y * nX + x] = mainHu[y * nX + x] = mainHv[y * nX + x] = 0;
    }

}

__kernel void variableMin_opencl_kernel(
        __global float *a,
        __global float *b
) {
    *a = min(*a, *b);
}

__kernel void variableSetInf_opencl_kernel(
        __global float *a
) {
    *a = INFINITY;
}

typedef enum BoundaryEdge {
    BND_LEFT, BND_RIGHT, BND_BOTTOM, BND_TOP
} BoundaryEdge;

enum RiemannState {
    DryDry = 0,
    WetWet = 1,
    WetDryInundation = 2,
    WetDryWall = 3,
    WetDryWallInundation = 4,
    DryWetInundation = 5,
    DryWetWall = 6,
    DryWetWallInundation = 7
};

enum RarefactionEnum {
    DrySingleRarefaction = 0,
    SingleRarefactionDry = 1,
    ShockShock = 2,
    ShockRarefaction = 3,
    RarefactionShock = 4,
    RarefactionRarefaction = 5

};


inline void
computeMiddleState(
        const real i_hLeft, const real i_hRight,
        const real i_uLeft, const real i_uRight,
        const real dryTol, const real newtonTol, const real g, const real sqrt_g,
        const unsigned int i_maxNumberOfNewtonIterations,
        real *o_hMiddle,
        real o_middleStateSpeeds[2]
) {
    //set everything to zero
    *o_hMiddle = (real)(0);
    o_middleStateSpeeds[0] = (real)(0);
    o_middleStateSpeeds[1] = (real)(0);

    //compute local square roots
    //(not necessarily the same ones as in computeNetUpdates!)
    real l_sqrt_g_hRight = sqrt(g * i_hRight);
    real l_sqrt_g_hLeft = sqrt(g * i_hLeft);

    //single rarefaction in the case of a wet/dry interface
    integer riemannStructure;
    if (i_hLeft < dryTol) {
        o_middleStateSpeeds[1] = o_middleStateSpeeds[0] = i_uRight - (real)(2) * l_sqrt_g_hRight;
        riemannStructure = DrySingleRarefaction;
        return;
    } else if (i_hRight < dryTol) {
        o_middleStateSpeeds[0] = o_middleStateSpeeds[1] = i_uLeft + (real)(2) * l_sqrt_g_hLeft;
        riemannStructure = SingleRarefactionDry;
        return;
    }

    //determine the wave structure of the Riemann-problem
/***************************************************************************************
 * Determine riemann structure begin
 **************************************************************************************/
    const real hMin = fmin(i_hLeft, i_hRight);
    const real hMax = fmax(i_hLeft, i_hRight);

    const real uDif = i_uRight - i_uLeft;

    if (0 <= (real)(2) * (sqrt(g * hMin) - sqrt(g * hMax)) + uDif) {
        riemannStructure = RarefactionRarefaction;
    } else if ((hMax - hMin) * sqrt(g * (real)(0.5) * (1 / hMax + 1 / hMin)) + uDif <= 0) {
        riemannStructure = ShockShock;
    } else if (i_hLeft < i_hRight) {
        riemannStructure = ShockRarefaction;
    } else {
        riemannStructure = RarefactionShock;
    }
/***************************************************************************************
 * Determine riemann structure end
 **************************************************************************************/

    //will be computed later
    real sqrt_g_hMiddle = (real)(0);

    if (riemannStructure == ShockShock) {
        *o_hMiddle = fmin(i_hLeft, i_hRight);

        real l_sqrtTermH[2] = {0, 0};

        for (unsigned int i = 0; i < i_maxNumberOfNewtonIterations; i++) {
            l_sqrtTermH[0] = sqrt((real)(0.5) * g * ((*o_hMiddle + i_hLeft) / (*o_hMiddle * i_hLeft)));
            l_sqrtTermH[1] = sqrt((real)(0.5) * g * ((*o_hMiddle + i_hRight) / (*o_hMiddle * i_hRight)));

            real phi = i_uRight - i_uLeft + (*o_hMiddle - i_hLeft) * l_sqrtTermH[0] +
                       (*o_hMiddle - i_hRight) * l_sqrtTermH[1];

            if (fabs(phi) < newtonTol) {
                break;
            }

            real derivativePhi = l_sqrtTermH[0] + l_sqrtTermH[1]
                                 - (real)(0.25) * g * (*o_hMiddle - i_hLeft) /
                                   (l_sqrtTermH[0] * *o_hMiddle * *o_hMiddle)
                                 - (real)(0.25) * g * (*o_hMiddle - i_hRight) /
                                   (l_sqrtTermH[1] * *o_hMiddle * *o_hMiddle);

            *o_hMiddle = *o_hMiddle - phi / derivativePhi; //Newton step
        }

        sqrt_g_hMiddle = sqrt(g * *o_hMiddle);
    }

    if (riemannStructure == RarefactionRarefaction) {
        *o_hMiddle = fmax((real)(0),
                         i_uLeft - i_uRight + (real)(2) * (l_sqrt_g_hLeft + l_sqrt_g_hRight));
        *o_hMiddle = (real)(1) / ((real)(16) * g) * *o_hMiddle * *o_hMiddle;

        sqrt_g_hMiddle = sqrt(g * *o_hMiddle);
    }

    if (riemannStructure == ShockRarefaction || riemannStructure == RarefactionShock) {
        real hMin, hMax;
        if (riemannStructure == ShockRarefaction) {
            hMin = i_hLeft;
            hMax = i_hRight;
        } else {
            hMin = i_hRight;
            hMax = i_hLeft;
        }

        *o_hMiddle = hMin;

        sqrt_g_hMiddle = sqrt(g * *o_hMiddle);
        real sqrt_g_hMax = sqrt(g * hMax);
        for (unsigned int i = 0; i < i_maxNumberOfNewtonIterations; i++) {
            real sqrtTermHMin = sqrt((real)(0.5) * g * ((*o_hMiddle + hMin) / (*o_hMiddle * hMin)));

            real phi = i_uRight - i_uLeft + (*o_hMiddle - hMin) * sqrtTermHMin +
                       (real)(2) * (sqrt_g_hMiddle - sqrt_g_hMax);

            if (fabs(phi) < newtonTol) {
                break;
            }

            real derivativePhi = sqrtTermHMin - (real)(0.25) * g * (*o_hMiddle - hMin) /
                                                (*o_hMiddle * *o_hMiddle * sqrtTermHMin) + sqrt_g / sqrt_g_hMiddle;

            *o_hMiddle = *o_hMiddle - phi / derivativePhi; //Newton step

            sqrt_g_hMiddle = sqrt(g * *o_hMiddle);
        }
    }

    o_middleStateSpeeds[0] = i_uLeft + (real)(2) * l_sqrt_g_hLeft - (real)(3) * sqrt_g_hMiddle;
    o_middleStateSpeeds[1] = i_uRight - (real)(2) * l_sqrt_g_hRight + (real)(3) * sqrt_g_hMiddle;

}

void waveSolver(
        const float_type i_hLeft, const float_type i_hRight,
        const float_type i_huLeft, const float_type i_huRight,
        const float_type i_bLeft, const float_type i_bRight,
        float_type o_netUpdates[5]
) {
    const float_type g = 9.81;
    const float_type dryTol = 0.01;
    const float_type newtonTol = 0.000001;
    const float_type zeroTol = 0.0001;
    const int maxNumberOfNewtonIterations = 10;

    real hLeft = i_hLeft;
    real hRight = i_hRight;
    real uLeft = (real)(0);
    real uRight = (real)(0);
    real huLeft = i_huLeft;
    real huRight = i_huRight;
    real bLeft = i_bLeft;
    real bRight = i_bRight;

    //declare variables which are used over and over again
    const real sqrt_g = sqrt(g);
    real sqrt_g_hLeft;
    real sqrt_g_hRight;

    real sqrt_hLeft;
    real sqrt_hRight;

    //set speeds to zero (will be determined later)
    uLeft = uRight = 0.;

    //reset net updates and the maximum wave speed
    o_netUpdates[0] = o_netUpdates[1] = o_netUpdates[2] = o_netUpdates[3] = (real)(0);
    o_netUpdates[4] = (real)(0);

    real hMiddle = (real)(0);
    real middleStateSpeeds[2] = {(real)(0)};

    //determine the wet/dry state and compute local variables correspondingly

/***************************************************************************************
 * Determine Wet Dry State Begin
 **************************************************************************************/
    integer wetDryState;
    //compute speeds or set them to zero (dry cells)
    if (hLeft > dryTol) {
        uLeft = huLeft / hLeft;
    } else {
        bLeft += hLeft;
        hLeft = huLeft = uLeft = 0;
    }

    if (hRight > dryTol) {
        uRight = huRight / hRight;
    } else {
        bRight += hRight;
        hRight = huRight = uRight = 0;
    }

    if (hLeft >= dryTol && hRight >= dryTol) {
        //test for simple wet/wet case since this is most probably the
        //most frequently executed branch
        wetDryState = WetWet;
    } else if (hLeft < dryTol && hRight < dryTol) {
        //check for the dry/dry-case
        wetDryState = DryDry;
    } else if (hLeft < dryTol && hRight + bRight > bLeft) {
        //we have a shoreline: one cell dry, one cell wet

        //check for simple inundation problems
        // (=> dry cell lies lower than the wet cell)
        wetDryState = DryWetInundation;
    } else if (hRight < dryTol && hLeft + bLeft > bRight) {
        wetDryState = WetDryInundation;
    } else if (hLeft < dryTol) {
        //dry cell lies higher than the wet cell
        //lets check if the momentum is able to overcome the difference in height
        //  => solve homogeneous Riemann-problem to determine the middle state height
        //     which would arise if there is a wall (wall-boundary-condition)
        //       \cite[ch. 6.8.2]{george2006finite})
        //       \cite[ch. 5.2]{george2008augmented}
        computeMiddleState(
                hRight, hRight,
                -uRight, uRight,
                dryTol, newtonTol, g, sqrt_g,
                maxNumberOfNewtonIterations,
                &hMiddle, middleStateSpeeds
        );

        if (hMiddle + bRight > bLeft) {
            //momentum is large enough, continue with the original values
            //          bLeft = o_hMiddle + bRight;
            wetDryState = DryWetWallInundation;
        } else {
            //momentum is not large enough, use wall-boundary-values
            hLeft = hRight;
            uLeft = -uRight;
            huLeft = -huRight;
            bLeft = bRight = (real)(0);
            wetDryState = DryWetWall;
        }
    } else if (hRight < dryTol) {
        //lets check if the momentum is able to overcome the difference in height
        //  => solve homogeneous Riemann-problem to determine the middle state height
        //     which would arise if there is a wall (wall-boundary-condition)
        //       \cite[ch. 6.8.2]{george2006finite})
        //       \cite[ch. 5.2]{george2008augmented}
        computeMiddleState(
                hLeft, hLeft,
                uLeft, -uLeft,
                dryTol, newtonTol, g, sqrt_g,
                maxNumberOfNewtonIterations,
                &hMiddle, middleStateSpeeds
        );

        if (hMiddle + bLeft > bRight) {
            //momentum is large enough, continue with the original values
            //          bRight = o_hMiddle + bLeft;
            wetDryState = WetDryWallInundation;
        } else {
            hRight = hLeft;
            uRight = -uLeft;
            huRight = -huLeft;
            bRight = bLeft = (real)(0);
            wetDryState = WetDryWall;
        }
    } else {
        //done with all cases
        //assert(false);
    }

    //limit the effect of the source term if there is a "wall"
    //\cite[end of ch. 5.2?]{george2008augmented}
    //\cite[rpn2ez_fast_geo.f][levequeclawpack]
    if (wetDryState == DryWetWallInundation) {
        bLeft = hRight + bRight;
    } else if (wetDryState == WetDryWallInundation) {
        bRight = hLeft + bLeft;
    }
/***************************************************************************************
 * Determine Wet Dry State End
 **************************************************************************************/

    if (wetDryState != DryDry) {
        //precompute some terms which are fixed during
        //the computation after some specific point
        sqrt_hLeft = sqrt(hLeft);
        sqrt_hRight = sqrt(hRight);

        sqrt_g_hLeft = sqrt_g * sqrt_hLeft;
        sqrt_g_hRight = sqrt_g * sqrt_hRight;


        //where to store the three waves
        real fWaves[3][2];
        //and their speeds
        real waveSpeeds[3];

        //compute the augmented decomposition
        //  (thats the place where the computational work is done..)
/***************************************************************************************
 * Compute Wave Decomposition Begin
 **************************************************************************************/
        //compute eigenvalues of the jacobian matrices in states Q_{i-1} and Q_{i} (char. speeds)
        const real characteristicSpeeds[2] = {
                uLeft - sqrt_g_hLeft,
                uRight + sqrt_g_hRight
        };

        //compute "Roe speeds"
        const real hRoe = (real)(0.5) * (hRight + hLeft);
        const real uRoe = (uLeft * sqrt_hLeft + uRight * sqrt_hRight) / (sqrt_hLeft + sqrt_hRight);

        //optimization for dumb compilers
        const real sqrt_g_hRoe = sqrt(g * hRoe);
        const real roeSpeeds[2] = {
                uRoe - sqrt_g_hRoe,
                uRoe + sqrt_g_hRoe
        };

        //compute the middle state of the homogeneous Riemann-Problem
        if (wetDryState != WetDryWall && wetDryState != DryWetWall) {
            //case WDW and DWW was computed in determineWetDryState already
            computeMiddleState(
                    hLeft, hRight,
                    uLeft, uRight,
                    dryTol, newtonTol, g, sqrt_g,
                    1,
                    &hMiddle, middleStateSpeeds
            );
        }

        //compute extended eindfeldt speeds (einfeldt speeds + middle state speeds)
        //  \cite[ch. 5.2]{george2008augmented}, \cite[ch. 6.8]{george2006finite}
        real extEinfeldtSpeeds[2] = {(real)(0), (real)(0)};
        if (wetDryState == WetWet || wetDryState == WetDryWall || wetDryState == DryWetWall) {
            extEinfeldtSpeeds[0] = fmin(characteristicSpeeds[0], roeSpeeds[0]);
            extEinfeldtSpeeds[0] = fmin(extEinfeldtSpeeds[0], middleStateSpeeds[1]);

            extEinfeldtSpeeds[1] = fmax(characteristicSpeeds[1], roeSpeeds[1]);
            extEinfeldtSpeeds[1] = fmax(extEinfeldtSpeeds[1], middleStateSpeeds[0]);
        } else if (hLeft < dryTol) {
            //ignore undefined speeds
            extEinfeldtSpeeds[0] = fmin(roeSpeeds[0], middleStateSpeeds[1]);
            extEinfeldtSpeeds[1] = fmax(characteristicSpeeds[1], roeSpeeds[1]);

        } else if (hRight < dryTol) {
            //ignore undefined speeds
            extEinfeldtSpeeds[0] = fmin(characteristicSpeeds[0], roeSpeeds[0]);
            extEinfeldtSpeeds[1] = fmax(roeSpeeds[1], middleStateSpeeds[0]);

        } else {
            //assert(false);
        }

        //HLL middle state
        //  \cite[theorem 3.1]{george2006finite}, \cite[ch. 4.1]{george2008augmented}
        const real hLLMiddleHeight = fmax(
                (huLeft - huRight + extEinfeldtSpeeds[1] * hRight - extEinfeldtSpeeds[0] * hLeft) /
                (extEinfeldtSpeeds[1] - extEinfeldtSpeeds[0]), (real)(0));

        //define eigenvalues
        const real eigenValues[3] = {
                extEinfeldtSpeeds[0],
                (real)(0.5) * (extEinfeldtSpeeds[0] + extEinfeldtSpeeds[1]),
                extEinfeldtSpeeds[1]
        };
        //define eigenvectors
        const real eigenVectors[3][3] = {
                {
                        (real)(1),
                        (real)(0),
                        (real)(1)
                },
                {
                        eigenValues[0],
                        (real)(0),
                        eigenValues[2]
                },
                {
                        eigenValues[0] * eigenValues[0],
                        (real)(1),
                        eigenValues[2] * eigenValues[2]
                }
        };


        //compute rarefaction corrector wave
        //  \cite[ch. 6.7.2]{george2006finite}, \cite[ch. 5.1]{george2008augmented}

        //compute the jump in state
        real rightHandSide[3] = {
                hRight - hLeft,
                huRight - huLeft,
                huRight * uRight + (real)(0.5) * g * hRight * hRight -
                (huLeft * uLeft + (real)(0.5) * g * hLeft * hLeft)
        };

        //compute steady state wave
        //  \cite[ch. 4.2.1 \& app. A]{george2008augmented}, \cite[ch. 6.2 \& ch. 4.4]{george2006finite}
        const real hBar = (hLeft + hRight) * (real)(0.5);

        real steadyStateWave[2] = {
                -(bRight - bLeft),
                -g * hBar * (bRight - bLeft)
        };

        //preserve depth-positivity
        //  \cite[ch. 6.5.2]{george2006finite}, \cite[ch. 4.2.3]{george2008augmented}
        if (eigenValues[0] < -zeroTol && eigenValues[2] > zeroTol) {
            //subsonic
            steadyStateWave[0] = fmax(steadyStateWave[0],
                                      hLLMiddleHeight * (eigenValues[2] - eigenValues[0]) / eigenValues[0]);
            steadyStateWave[0] = fmin(steadyStateWave[0],
                                      hLLMiddleHeight * (eigenValues[2] - eigenValues[0]) / eigenValues[2]);
        } else if (eigenValues[0] > zeroTol) {
            //supersonic right TODO: motivation?
            steadyStateWave[0] = fmax(steadyStateWave[0], -hLeft);
            steadyStateWave[0] = fmin(steadyStateWave[0],
                                      hLLMiddleHeight * (eigenValues[2] - eigenValues[0]) / eigenValues[0]);
        } else if (eigenValues[2] < -zeroTol) {
            //supersonic left TODO: motivation?
            steadyStateWave[0] = fmax(steadyStateWave[0],
                                      hLLMiddleHeight * (eigenValues[2] - eigenValues[0]) / eigenValues[2]);
            steadyStateWave[0] = fmin(steadyStateWave[0], hRight);
        }

        //Limit the effect of the source term
        //  \cite[ch. 6.4.2]{george2006finite}
        steadyStateWave[1] = fmin(steadyStateWave[1], g * fmax(-hLeft * (bRight - bLeft), -hRight * (bRight - bLeft)));
        steadyStateWave[1] = fmax(steadyStateWave[1], g * fmin(-hLeft * (bRight - bLeft), -hRight * (bRight - bLeft)));

        rightHandSide[0] -= steadyStateWave[0];
        //rightHandSide[1]: no source term
        rightHandSide[2] -= steadyStateWave[1];

        //everything is ready, solve the equations!
/***************************************************************************************
 * Solve linear equation begin
 **************************************************************************************/
        // compute inverse of 3x3 matrix
        const real m[3][3] = {
                {
                        (eigenVectors[1][1] * eigenVectors[2][2] - eigenVectors[1][2] * eigenVectors[2][1]),
                        -(eigenVectors[0][1] * eigenVectors[2][2] - eigenVectors[0][2] * eigenVectors[2][1]),
                        (eigenVectors[0][1] * eigenVectors[1][2] - eigenVectors[0][2] * eigenVectors[1][1])
                },
                {
                        -(eigenVectors[1][0] * eigenVectors[2][2] - eigenVectors[1][2] * eigenVectors[2][0]),
                        (eigenVectors[0][0] * eigenVectors[2][2] - eigenVectors[0][2] * eigenVectors[2][0]),
                        -(eigenVectors[0][0] * eigenVectors[1][2] - eigenVectors[0][2] * eigenVectors[1][0])
                },
                {
                        (eigenVectors[1][0] * eigenVectors[2][1] - eigenVectors[1][1] * eigenVectors[2][0]),
                        -(eigenVectors[0][0] * eigenVectors[2][1] - eigenVectors[0][1] * eigenVectors[2][0]),
                        (eigenVectors[0][0] * eigenVectors[1][1] - eigenVectors[0][1] * eigenVectors[1][0])
                }
        };
        const real d = (eigenVectors[0][0] * m[0][0] + eigenVectors[0][1] * m[1][0] + eigenVectors[0][2] * m[2][0]);

        // m stores not really the inverse matrix, but the inverse multiplied by d
        const real s = 1 / d;

        // compute m*rightHandSide
        const real beta[3] = {
                (m[0][0] * rightHandSide[0] + m[0][1] * rightHandSide[1] + m[0][2] * rightHandSide[2]) * s,
                (m[1][0] * rightHandSide[0] + m[1][1] * rightHandSide[1] + m[1][2] * rightHandSide[2]) * s,
                (m[2][0] * rightHandSide[0] + m[2][1] * rightHandSide[1] + m[2][2] * rightHandSide[2]) * s
        };
/***************************************************************************************
 * Solve linear equation end
 **************************************************************************************/

        //compute f-waves and wave-speeds
        if (wetDryState == WetDryWall) {
            //zero ghost updates (wall boundary)
            //care about the left going wave (0) only
            fWaves[0][0] = beta[0] * eigenVectors[1][0];
            fWaves[0][1] = beta[0] * eigenVectors[2][0];

            //set the rest to zero
            fWaves[1][0] = fWaves[1][1] = (real)(0);
            fWaves[2][0] = fWaves[2][1] = (real)(0);

            waveSpeeds[0] = eigenValues[0];
            waveSpeeds[1] = waveSpeeds[2] = (real)(0);

        } else if (wetDryState == DryWetWall) {
            //zero ghost updates (wall boundary)
            //care about the right going wave (2) only
            fWaves[2][0] = beta[2] * eigenVectors[1][2];
            fWaves[2][1] = beta[2] * eigenVectors[2][2];

            //set the rest to zero
            fWaves[0][0] = fWaves[0][1] = (real)(0);
            fWaves[1][0] = fWaves[1][1] = (real)(0);

            waveSpeeds[2] = eigenValues[2];
            waveSpeeds[0] = waveSpeeds[1] = (real)(0);

        } else {
            //compute f-waves (default)
            for (int waveNumber = 0; waveNumber < 3; waveNumber++) {
                fWaves[waveNumber][0] = beta[waveNumber] * eigenVectors[1][waveNumber]; //select 2nd and
                fWaves[waveNumber][1] =
                        beta[waveNumber] * eigenVectors[2][waveNumber]; //3rd component of the augmented decomposition
            }

            waveSpeeds[0] = eigenValues[0];
            waveSpeeds[1] = eigenValues[1];
            waveSpeeds[2] = eigenValues[2];
        }
/***************************************************************************************
 * Compute Wave Decomposition End
 **************************************************************************************/


        //compute the updates from the three propagating waves
        //A^-\delta Q = \sum{s[i]<0} \beta[i] * r[i] = A^-\delta Q = \sum{s[i]<0} Z^i
        //A^+\delta Q = \sum{s[i]>0} \beta[i] * r[i] = A^-\delta Q = \sum{s[i]<0} Z^i
        for (int waveNumber = 0; waveNumber < 3; waveNumber++) {
            if (waveSpeeds[waveNumber] < -zeroTol) {
                //left going
                o_netUpdates[0] += fWaves[waveNumber][0];
                o_netUpdates[2] += fWaves[waveNumber][1];
            } else if (waveSpeeds[waveNumber] > zeroTol) {
                //right going
                o_netUpdates[1] += fWaves[waveNumber][0];
                o_netUpdates[3] += fWaves[waveNumber][1];
            } else {
                //TODO: this case should not happen mathematically, but it does. Where is the bug? Machine accuracy only?
                o_netUpdates[0] += (real)(0.5) * fWaves[waveNumber][0];
                o_netUpdates[2] += (real)(0.5) * fWaves[waveNumber][1];

                o_netUpdates[1] += (real)(0.5) * fWaves[waveNumber][0];
                o_netUpdates[3] += (real)(0.5) * fWaves[waveNumber][1];
            }
        }

        //compute maximum wave speed (-> CFL-condition)
        waveSpeeds[0] = fabs(waveSpeeds[0]);
        waveSpeeds[1] = fabs(waveSpeeds[1]);
        waveSpeeds[2] = fabs(waveSpeeds[2]);

        o_netUpdates[4] = fmax(waveSpeeds[0], waveSpeeds[1]);
        o_netUpdates[4] = fmax(o_netUpdates[4], waveSpeeds[2]);
    }
}

//https://stackoverflow.com/questions/18950732/atomic-max-for-floats-in-opencl
//Function to perform the atomic max
inline void atomicMin(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = min(prevVal.floatVal, operand);
    } while (atomic_cmpxchg((volatile __global unsigned int *) source, prevVal.intVal, newVal.intVal) !=
             prevVal.intVal);
}

__kernel void computeNumericalFluxes_border_opencl_kernel(
        __global float_type *hMain,
        __global float_type *huMain,
        __global float_type *hvMain,
        __global float_type *hNeighbour,
        __global float_type *huNeighbour,
        __global float_type *hvNeighbour,
        __global float_type *b,
        __global float *maxTimestep,
        __global float_type *hNetUpdates,
        __global float_type *huNetUpdates,
        __global float_type *hvNetUpdates,
        const int nX,
        const int nY,
        const float dX_inv,
        const float dY_inv,
        const float dX,
        const float dY,
        const BoundaryEdge side,
        __local float *localMaxEdgeSpeed
) {
    const unsigned int index = get_global_id(0);
    const unsigned int lidx = get_local_id(0);
    localMaxEdgeSpeed[lidx] = 0;
    const unsigned int n = side == BND_LEFT || side == BND_RIGHT ? nY : nX;
    if (index < n) {
        if (side == BND_RIGHT || side == BND_LEFT) {

            float_type hLeft = (side == BND_LEFT ? hNeighbour : hMain)[
                    index * nX + (side == BND_LEFT ? 0 : (nX - 1))
            ];
            float_type hRight = (side == BND_LEFT ? hNeighbour : hMain)[
                    index
            ];
            float_type huLeft = (side == BND_LEFT ? huNeighbour : huMain)[
                    index * nX + (side == BND_LEFT ? 0 : (nX - 1))
            ];
            float_type huRight = (side == BND_LEFT ? huNeighbour : huMain)[
                    index
            ];
            float_type bLeft = b[
                    (index + 1) * (nX + 2) +
                    (side == BND_LEFT ? 0 : nX)];
            float_type bRight = b[(index + 1) * (nX + 2) + 1 +
                                  (side == BND_LEFT ? 0 : nX)];

            float_type updates[5];

            waveSolver(
                    hLeft, hRight,
                    huLeft, huRight,
                    bLeft, bRight,
                    updates
            );
            localMaxEdgeSpeed[lidx] = fmax(localMaxEdgeSpeed[lidx], updates[4]);

            if (side == BND_LEFT) {
                hNetUpdates[index] += dX_inv * updates[1];
                huNetUpdates[index] += dX_inv * updates[3];
            } else {
                hNetUpdates[index * nX + nX - 1] += dX_inv * updates[0];
                huNetUpdates[index * nX + nX - 1] += dX_inv * updates[2];
            }
        } else {

            float_type hUpper = (side == BND_TOP ? hNeighbour : hMain)[(side == BND_TOP ? 0 : (nY - 1)) * nX +
                                                                       index + (side == BND_TOP ? 1 : 0)];
            float_type hLower = (side == BND_TOP ? hMain : hNeighbour)[side == BND_TOP ? index : (index + 1)];
            float_type hvUpper = (side == BND_TOP ? hvNeighbour : hvMain)[
                    (side == BND_TOP ? 0 : (nY - 1)) * nX + index + (side == BND_TOP ? 1 : 0)];
            float_type hvLower = (side == BND_TOP ? hvMain : hvNeighbour)[side == BND_TOP ? index : (index + 1)];

            float_type bUpper = b[(side == BND_TOP ? 0 : nY) *
                                  (nX + 2) + (index + 1)];
            float_type bLower = b[((side == BND_TOP ? 0 : nY) + 1) *
                                  (nX + 2) + (index + 1)];

            float_type updates[5];
            waveSolver(
                    hUpper, hLower,
                    hvUpper, hvLower,
                    bUpper, bLower,
                    updates
            );
            localMaxEdgeSpeed[lidx] = fmax(localMaxEdgeSpeed[lidx], updates[4]);

            if (side == BND_TOP) {
                hNetUpdates[index * nX] += dY_inv * updates[1];
                hvNetUpdates[index * nX] += dY_inv * updates[3];
            } else {
                hNetUpdates[index * nX + nY - 1] += dY_inv * updates[0];
                hvNetUpdates[index * nX + nY - 1] += dY_inv * updates[2];
            }
        }
    }

    const int group_size = get_local_size(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int s = group_size / 2; s > 0; s >>= 1) {
        if (lidx < s && lidx + s < group_size) {
            localMaxEdgeSpeed[lidx] = fmax(localMaxEdgeSpeed[lidx], localMaxEdgeSpeed[lidx + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lidx == 0) {
        float localMaxTimeStep = FLT_MAX;
        if (localMaxEdgeSpeed[0] > 0) {
            localMaxTimeStep = fmin(dX / localMaxEdgeSpeed[0], dY / localMaxEdgeSpeed[0]) * 0.4;
        }
        atomicMin(maxTimestep, localMaxTimeStep);
    }
}

__kernel void computeNumericalFluxes_mainBlock_opencl_kernel(
        __global float_type *hMain,
        __global float_type *huMain,
        __global float_type *hvMain,
        __global float_type *b,
        __global float *maxTimestep,
        __global float_type *hNetUpdates,
        __global float_type *huNetUpdates,
        __global float_type *hvNetUpdates,
        const unsigned int nX,
        const unsigned int nY,
        const float dX_inv,
        const float dY_inv,
        const float dX,
        const float dY,
        const BoundaryEdge side,
        __local float *localMaxEdgeSpeed
) {
    const unsigned int idxX = get_global_id(0);
    const unsigned int idxY = get_global_id(1);

    const unsigned int threadIdxLin = get_local_id(1)*get_local_size(0)+get_local_id(0);
    localMaxEdgeSpeed[threadIdxLin] = 0;

    if (idxX > 0 && idxX < nX && idxY < nY) {

        float_type updates[5];

        float_type hLeft = hMain[idxY*nX+idxX - 1];
        float_type hRight = hMain[idxY*nX+idxX];
        float_type huLeft = huMain[idxY*nX+idxX - 1];
        float_type huRight = huMain[idxY*nX+idxX];

        float_type bLeft = b[(idxY + 1) * (nX+2) + idxX];
        float_type bRight = b[(idxY + 1) * (nX+2) + idxX + 1];
        waveSolver(
                hLeft, hRight,
                huLeft, huRight,
                bLeft, bRight,
                updates
        );
        localMaxEdgeSpeed[threadIdxLin] = fmax(localMaxEdgeSpeed[threadIdxLin], updates[4]);

        hNetUpdates[idxY*nX+idxX-1] += dX_inv * updates[0];
        hNetUpdates[idxY*nX+idxX] += dX_inv * updates[1];
        huNetUpdates[idxY*nX+idxX-1] += dX_inv * updates[2];
        huNetUpdates[idxY*nX+idxX] += dX_inv * updates[3];
    }
    if (idxX < nX && idxY < nY - 1) {
        float_type updates[5];

        float_type hUpper = hMain[idxY*nX+idxX];
        float_type hLower = hMain[(idxY+1)*nX+idxX];
        float_type hvUpper = hvMain[idxY*nX+idxX - 1];
        float_type hvLower = hvMain[(idxY+1)*nX+idxX];

        float_type bUpper = b[(idxY + 1) * (nX+2) + idxX+1];
        float_type bLower = b[(idxY + 2) * (nX+2) + idxX + 1];
        waveSolver(
                hUpper, hLower,
                hvUpper, hvLower,
                bUpper, bLower,
                updates
        );
        localMaxEdgeSpeed[threadIdxLin] = fmax(localMaxEdgeSpeed[threadIdxLin], updates[4]);

        hNetUpdates[idxY*nX+idxX] += dY_inv * updates[0];
        hNetUpdates[(idxY+1)*nX+idxX] += dY_inv * updates[1];
        huNetUpdates[idxY*nX+idxX] += dY_inv * updates[2];
        huNetUpdates[(idxY+1)*nX+idxX] += dY_inv * updates[3];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int group_size = get_local_size(0)*get_local_size(1);

    for (unsigned int s = group_size / 2; s > 0; s >>= 1) {
        if (threadIdxLin < s && threadIdxLin + s < group_size) {
            localMaxEdgeSpeed[threadIdxLin] = fmax(localMaxEdgeSpeed[threadIdxLin], localMaxEdgeSpeed[threadIdxLin + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (threadIdxLin == 0) {
        float localMaxTimeStep = FLT_MAX;
        if (localMaxEdgeSpeed[0] > 0) {
            localMaxTimeStep = fmin(dX / localMaxEdgeSpeed[0], dY / localMaxEdgeSpeed[0]) * 0.4;
        }
        atomicMin(maxTimestep, localMaxTimeStep);
    }
}

//OpenCL is so stupid, it doesn't even have memset
__kernel void memset(__global float* p, const float value, const unsigned long n){
    const unsigned long i = get_global_id(0);
    if(i <n){
        p[i] = value;
    }
}