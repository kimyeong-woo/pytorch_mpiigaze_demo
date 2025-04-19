import sys

sys.path.append('./pytorch_mpiigaze_demo/ptgaze')

import ptgaze
from ptgaze.main import main

if __name__ == '__main__':
    # Set argparse to parse command line arguments

    # sys.argv.append('--config')
    # sys.argv.append('./pytorch_mpiigaze_demo/configs/mpiigaze.yaml')

    sys.argv.append('--mode')
    sys.argv.append('mpiigaze')
    # sys.argv.append('mpiifacegaze')
    # sys.argv.append('eth-xgaze')

    sys.argv.append('--face-detector')
    # sys.argv.append('dlib')
    # sys.argv.append('face_alignment_dlib')
    # sys.argv.append('face_alignment_sfd')
    sys.argv.append('mediapipe')

    sys.argv.append('--device')
    # sys.argv.append('cuda')
    sys.argv.append('cpu')

    # sys.argv.append('--image')
    # sys.argv.append('./pytorch_mpiigaze_demo/data/images/face.jpg')

    # sys.argv.append('--video')
    # sys.argv.append('./pytorch_mpiigaze_demo/data/videos/face.mp4')

    # sys.argv.append('--camera')
    # sys.argv.append('./pytorch_mpiigaze_demo/data/calib/sample_params.yaml')

    # sys.argv.append('--output-dir')
    # sys.argv.append('./pytorch_mpiigaze_demo/data/output')

    # sys.argv.append('--ext')
    # sys.argv.append('avi')
    # sys.argv.append('mp4')

    # sys.argv.append('--no-screen')

    # sys.argv.append('--debug')

    main()


