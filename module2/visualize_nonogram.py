import numpy as np

DISPLAY_MODE = True
PRINTING_PROGRESSION = False
DISPLAY_SPEED = 0.3		  # seconds between each update of the visualization

BOARD_SIZE = 10
IMAGE = None

if DISPLAY_MODE or PRINTING_PROGRESSION:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Nonogram')


board = [[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
         [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], ]

IMAGE.set_data(board)
plt.pause(DISPLAY_SPEED)
plt.show()      # stops image from disappearing after the short pause
