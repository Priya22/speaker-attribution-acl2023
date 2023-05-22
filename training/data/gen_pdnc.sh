#!/bin/bash
. /etc/profile.d/lmod.sh
module use /pkgs/environment-modules/
python /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/gen_training_pdnc.py explicit /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/data/pdnc/explicit
# python /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/gen_training_pdnc.py random /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/data/pdnc/random
# python /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/gen_training_pdnc.py loo /h/vkpriya/bookNLP/booknlp-en/training/data/speaker_attribution/data/pdnc/loo
