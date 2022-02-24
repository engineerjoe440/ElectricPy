################################################################################
"""
`electricpy.constants` - Electrical Engineering Constants.

Defenition of all required constants and matricies for
*electricpy* module.
"""
################################################################################

# Import Supporting Dependencies
import numpy as _np
import cmath as _c

# Define Electrical Engineering Constants
pi = _np.pi  #: PI Constant 3.14159...
a = _c.rect(1, _np.radians(120))  #: 'A' Operator for Symmetrical Components
p = 1e-12  #: Pico Multiple      (10^-12)
n = 1e-9  #: Nano Multiple       (10^-9)
u = 1e-6  #: Micro (mu) Multiple (10^-6)
m = 1e-3  #: Mili Multiple       (10^-3)
k = 1e3  #: Kili Multiple       (10^3)
M = 1e6  #: Mega Multiple       (10^6)
G = 1e9  #: Giga Multiple       (10^9)
u0 = 4 * _np.pi * 10 ** (-7)  #: µ0 (mu-not)       4πE-7
e0 = 8.8541878128e-12  #: ε0 (epsilon-not)  8.854E-12
carson_r = 9.869e-7  #: Carson's Ristance Constant  8.869E-7
De0 = 2160  #: De Constant for Use with Transmission Impedance Calculations      2160
NAN = float("nan")
VLLcVLN = _c.rect(_np.sqrt(3), _np.radians(30))  # Conversion Operator
ILcIP = _c.rect(_np.sqrt(3), _np.radians(30))  # Conversion Operator
WATTS_PER_HP = 745.699872
KWH_PER_BTU = 3412.14

# Define Symmetrical Component Matricies
Aabc = (
    1
    / 3
    * _np.array(
        [
            [1, 1, 1],  # Convert ABC to 012
            [1, a, a**2],  # (i.e. phase to sequence)
            [1, a**2, a],
        ]
    )
)
A012 = _np.array(
    [
        [1, 1, 1],  # Convert 012 to ABC
        [1, a**2, a],  # (i.e. sequence to phase)
        [1, a, a**2],
    ]
)
# Define Clarke Component Matricies
Cabc = _np.sqrt(2 / 3) * _np.array(
    [
        [1, -1 / 2, -1 / 2],  # Convert ABC to alpha/beta/gamma
        [0, _np.sqrt(3) / 2, -_np.sqrt(3) / 2],
        [1 / _np.sqrt(2), 1 / _np.sqrt(2), 1 / _np.sqrt(2)],
    ]
)
Cxyz = _np.array(
    [
        [2 / _np.sqrt(6), 0, 1 / _np.sqrt(3)],  # Convert alpha/beta/gamma to ABC
        [-1 / _np.sqrt(6), 1 / _np.sqrt(2), 1 / _np.sqrt(3)],
        [-1 / _np.sqrt(6), -1 / _np.sqrt(2), 1 / _np.sqrt(3)],
    ]
)
# Define Park Components Matricies
_rad = lambda th: _np.radians(th)
Pdq0_im = lambda th: _np.sqrt(2 / 3) * _np.array(
    [
        [
            _np.cos(_rad(th)),
            _np.cos(_rad(th) - 2 * pi / 3),
            _np.cos(_rad(th) + 2 * pi / 3),
        ],
        [
            -_np.sin(_rad(th)),
            -_np.sin(_rad(th) - 2 * pi / 3),
            -_np.sin(_rad(th) + 2 * pi / 3),
        ],
        [_np.sqrt(2) / 2, _np.sqrt(2) / 2, _np.sqrt(2) / 2],
    ]
)
Pabc_im = lambda th: _np.sqrt(2 / 3) * _np.array(
    [
        [_np.cos(_rad(th)), -_np.sin(_rad(th)), _np.sqrt(2) / 2],
        [
            _np.cos(_rad(th) - 2 * pi / 3),
            -_np.sin(_rad(th) - 2 * pi / 3),
            _np.sqrt(2) / 2,
        ],
        [
            _np.cos(_rad(th) + 2 * pi / 3),
            -_np.sin(_rad(th) + 2 * pi / 3),
            _np.sqrt(2) / 2,
        ],
    ]
)
Pdq0 = (
    2
    / 3
    * _np.array(
        [
            [0, -_np.sqrt(3 / 2), _np.sqrt(3 / 2)],
            [1, -1 / 2, -1 / 2],
            [1 / 2, 1 / 2, 1 / 2],
        ]
    )
)
Pqd0 = (
    2
    / 3
    * _np.array(
        [
            [1, -1 / 2, -1 / 2],
            [0, -_np.sqrt(3 / 2), _np.sqrt(3 / 2)],
            [1 / 2, 1 / 2, 1 / 2],
        ]
    )
)

# Define Transformer Shift Correction Matricies
XFMY0 = _np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
XFMD1 = 1 / _np.sqrt(3) * _np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]])
XFMD11 = 1 / _np.sqrt(3) * _np.array([[1, 0, -1], [-1, 1, 0], [0, -1, 1]])
XFM12 = 1 / 3 * _np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

# Define Complex Angle Terms
e30 = _c.rect(1, _np.radians(30))  #: 30° Phase Operator
en30 = _c.rect(1, _np.radians(-30))  #: -30° Phase Operator
e60 = _c.rect(1, _np.radians(60))  #: 60° Phase Operator
en60 = _c.rect(1, _np.radians(-60))  #: -60° Phase Operator
e90 = _c.rect(1, _np.radians(90))  #: 90° Phase Operator
en90 = _c.rect(1, _np.radians(-90))  #: -90° Phase Operator
e45 = _c.rect(1, _np.radians(45))  #: 45° Phase Operator
en45 = _c.rect(1, _np.radians(-45))  #: -45° Phase Operator

# Define Material Resistivity (Rho)
resistivity_rho = {
    "silver": 15.9,
    "copper": 16.8,
    "aluminium": 6.5,
    "tungsten": 56,
    "iron": 97.1,
    "platinum": 106,
    "manganin": 482,
    "lead": 220,
    "mercury": 980,
    "nichrome": 1000,
    "constantan": 490,
}
THERMO_COUPLE_DATA = {
    "J": [
        [-6.4936529e01, 2.5066947e02, 6.4950262e02, 9.2510550e02, 1.0511294e03],
        [-3.1169773e00, 1.3592329e01, 3.6040848e01, 5.3433832e01, 6.0956091e01],
        [2.2133797e01, 1.8014787e01, 1.6593395e01, 1.6243326e01, 1.7156001e01],
        [2.0476437e00, -6.5218881e-02, 7.3009590e-01, 9.2793267e-01, -2.5931041e00],
        [-4.6867532e-01, -1.2179108e-02, 2.4157343e-02, 6.4644193e-03, -5.8339803e-02],
        [-3.6673992e-02, 2.0061707e-04, 1.2787077e-03, 2.0464414e-03, 1.9954137e-02],
        [1.1746348e-01, -3.9494552e-03, 4.9172861e-02, 5.2541788e-02, -1.5305581e-01],
        [-2.0903413e-02, -7.3728206e-04, 1.6813810e-03, 1.3682959e-04, -2.9523967e-03],
        [-2.1823704e-03, 1.6679731e-05, 7.6067922e-05, 1.3454746e-04, 1.1340164e-03],
    ],
    "K": [
        [-1.2147164e02, -8.7935962e00, 3.1018976e02, 6.0572562e02, 1.0184705e03],
        [-4.1790858e00, -3.4489914e-01, 1.2631386e01, 2.5148718e01, 4.1993851e01],
        [3.6069513e01, 2.5678719e01, 2.4061949e01, 2.3539401e01, 2.5783239e01],
        [3.0722076e01, -4.9887904e-01, 4.0158622e00, 4.6547228e-02, -1.8363403e00],
        [7.7913860e00, -4.4705222e-01, 2.6853917e-01, 1.3444400e-02, 5.6176662e-02],
        [5.2593991e-01, -4.4869203e-02, -9.7188544e-03, 5.9236853e-04, 1.8532400e-04],
        [9.3939547e-01, 2.3893439e-04, 1.6995872e-01, 8.3445513e-04, -7.4803355e-02],
        [2.7791285e-01, -2.0397750e-02, 1.1413069e-02, 4.6121445e-04, 2.3841860e-03],
        [2.5163349e-02, -1.8424107e-03, -3.9275155e-04, 2.5488122e-05, 0.0],
    ],
    "B": [
        [5.0000000e02, 1.2461474e03],
        [1.2417900e00, 7.2701221e00],
        [1.9858097e02, 9.4321033e01],
        [2.4284248e01, 7.3899296e00],
        [-9.7271640e01, -1.5880987e-01],
        [-1.5701178e01, 1.2681877e-02],
        [3.1009445e-01, 1.0113834e-01],
        [-5.0880251e-01, -1.6145962e-03],
        [-1.6163342e-01, -4.1086314e-06],
    ],
    "E": [
        [-1.1721668e02, -5.0000000e01, 2.5014600e02, 6.0139890e02, 8.0435911e02],
        [-5.9901698e00, -2.7871777e00, 1.7191713e01, 4.5206167e01, 6.1359178e01],
        [2.3647275e01, 1.9022736e01, 1.3115522e01, 1.2399357e01, 1.2759508e01],
        [1.2807377e01, -1.7042725e00, 1.1780364e00, 4.3399963e-01, -1.1116072e00],
        [2.0665069e00, -3.5195189e-01, 3.6422433e-02, 9.1967085e-03, 3.5332536e-02],
        [8.6513472e-02, 4.7766102e-03, 3.9584261e-04, 1.6901585e-04, 3.3080380e-05],
        [5.8995860e-01, -6.5379760e-02, 9.3112756e-02, 3.4424680e-02, -8.8196889e-02],
        [1.0960713e-01, -2.1732833e-02, 2.9804232e-03, 6.9741215e-04, 2.8497415e-03],
        [6.1769588e-03, 0.0, 3.3263032e-05, 1.2946992e-05, 0.0],
    ],
    "N": [
        [-5.9610511e01, 3.1534505e02, 1.0340172e03],
        [-1.5000000e00, 9.8870997e00, 3.7565475e01],
        [4.2021322e01, 2.7988676e01, 2.6029492e01],
        [4.7244037e00, 1.5417343e00, -6.0783095e-01],
        [-6.1153213e00, -1.4689457e-01, -9.7742562e-03],
        [-9.9980337e-01, -6.8322712e-03, -3.3148813e-06],
        [1.6385664e-01, 6.2600036e-02, -2.5351881e-02],
        [-1.4994026e-01, -5.1489572e-03, -3.8746827e-04],
        [-3.0810372e-02, -2.8835863e-04, 1.7088177e-06],
    ],
    "R": [
        [1.3054315e02, 5.4188181e02, 1.0382132e03, 1.5676133e03],
        [8.8333090e-01, 4.9312886e00, 1.1014763e01, 1.8397910e01],
        [1.2557377e02, 9.0208190e01, 7.4669343e01, 7.1646299e01],
        [1.3900275e02, 6.1762254e00, 3.4090711e00, -1.0866763e00],
        [3.3035469e01, -1.2279323e00, -1.4511205e-01, -2.0968371e00],
        [-8.5195924e-01, 1.4873153e-02, 6.3077387e-03, -7.6741168e-01],
        [1.2232896e00, 8.7670455e-02, 5.6880253e-02, -1.9712341e-02],
        [3.5603023e-01, -1.2906694e-02, -2.0512736e-03, -2.9903595e-02],
        [0.0, 0.0, 0.0, -1.0766878e-02],
    ],
    "S": [
        [1.3792630e02, 4.7673468e02, 9.7946589e02, 1.6010461e03],
        [9.3395024e-01, 4.0037367e00, 9.3508283e00, 1.6789315e01],
        [1.2761836e02, 1.0174512e02, 8.7126730e01, 8.4315871e01],
        [1.1089050e02, -8.9306371e00, -2.3139202e00, -1.0185043e01],
        [1.9898457e01, -4.2942435e00, -3.2682118e-02, -4.6283954e00],
        [9.6152996e-02, 2.0453847e-01, 4.6090022e-03, -1.0158749e00],
        [9.6545918e-01, -7.1227776e-02, -1.4299790e-02, -1.2877783e-01],
        [2.0813850e-01, -4.4618306e-02, -1.2289882e-03, -5.5802216e-02],
        [0.0, 1.6822887e-03, 0.0, -1.2146518e-02],
    ],
    "T": [
        [-1.9243000e02, -6.0000000e01, 1.3500000e02, 3.0000000e02],
        [-5.4798963e00, -2.1528350e00, 5.9588600e00, 1.4861780e01],
        [5.9572141e01, 3.0449332e01, 2.0325591e01, 1.7214707e01],
        [1.9675733e00, -1.2946560e00, 3.3013079e00, -9.3862713e-01],
        [-7.8176011e01, -3.0500735e00, 1.2638462e-01, -7.3509066e-02],
        [-1.0963280e01, -1.9226856e-01, -8.2883695e-04, 2.9576140e-04],
        [2.7498092e-01, 6.9877863e-03, 1.7595577e-01, -4.8095795e-02],
        [-1.3768944e00, -1.0596207e-01, 7.9740521e-03, -4.7352054e-03],
        [-4.5209805e-01, -1.0774995e-02, 0.0, 0.0],
    ],
}
THERMO_COUPLE_KEYS = ["To", "Vo", "P1", "P2", "P3", "P4", "Q1", "Q2", "Q3"]
THERMO_COUPLE_VOLTAGES = {
    "J": [-8.095, 0, 21.840, 45.494, 57.953, 69.553],
    "K": [-6.404, -3.554, 4.096, 16.397, 33.275, 69.553],
    "B": [0.291, 2.431, 13.820, None, None, None],
    "E": [-9.835, -5.237, 0.591, 24.964, 53.112, 76.373],
    "N": [-4.313, 0, 20.613, 47.513, None, None],
    "R": [-0.226, 1.469, 7.461, 14.277, 21.101, None],
    "S": [-0.236, 1.441, 6.913, 12.856, 18.693, None],
    "T": [-6.18, -4.648, 0, 9.288, 20.872, None],
}

COLD_JUNCTION_DATA = {
    "To": [
        4.2000000e01,
        2.5000000e01,
        2.5000000e01,
        2.5000000e01,
        7.0000000e00,
        2.5000000e01,
        2.5000000e01,
        2.5000000e01,
    ],
    "Vo": [
        3.3933898e-04,
        1.4950582e00,
        1.2773432e00,
        1.0003453e00,
        1.8210024e-01,
        1.4067016e-01,
        1.4269163e-01,
        9.9198279e-01,
    ],
    "P1": [
        2.1196684e-04,
        6.0958443e-02,
        5.1744084e-02,
        4.0514854e-02,
        2.6228256e-02,
        5.9330356e-03,
        5.9829057e-03,
        4.0716564e-02,
    ],
    "P2": [
        3.3801250e-06,
        -2.7351789e-04,
        -5.4138663e-05,
        -3.8789638e-05,
        -1.5485539e-04,
        2.7736904e-05,
        4.5292259e-06,
        7.1170297e-04,
    ],
    "P3": [
        -1.4793289e-07,
        -1.9130146e-05,
        -2.2895769e-06,
        -2.8608478e-06,
        2.1366031e-06,
        -1.0819644e-06,
        -1.3380281e-06,
        6.8782631e-07,
    ],
    "P4": [
        -3.3571424e-09,
        -1.3948840e-08,
        -7.7947143e-10,
        -9.5367041e-10,
        9.2047105e-10,
        -2.3098349e-09,
        -2.3742577e-09,
        4.3295061e-11,
    ],
    "Q1": [
        -1.0920410e-02,
        -5.2382378e-03,
        -1.5173342e-03,
        -1.3948675e-03,
        -6.4070932e-03,
        2.6146871e-03,
        -1.0650446e-03,
        1.6458102e-02,
    ],
    "Q2": [
        -4.9782932e-04,
        -3.0970168e-04,
        -4.2314514e-05,
        -6.7976627e-05,
        8.2161781e-05,
        -1.8621487e-04,
        -2.2042420e-04,
        0.0000000e00,
    ],
}
COLD_JUNCTION_KEYS = ["To", "Vo", "P1", "P2", "P3", "P4", "Q1", "Q2"]

RTD_TYPES = {
    "PT100": [100, 0.00385],
    "PT1000": [1000, 0.00385],
    "CU100": [100, 0.00427],
    "NI100": [100, 0.00618],
    "NI120": [120, 0.00672],
    "NIFE": [604, 0.00518],
}

RHO_VALUES = {
    "SEA": 0.01,
    "SWAMP": 10,
    "AVG": 100,
    "AVERAGE": 100,
    "DAMP": 100,
    "DRY": 1000,
    "SAND": 1e9,
    "SANDSTONE": 1e9,
}
# END OF FILE
