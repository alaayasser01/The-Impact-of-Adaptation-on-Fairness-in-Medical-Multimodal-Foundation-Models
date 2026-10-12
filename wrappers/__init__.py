from .clip import CLIPWrapper
from .linear_probe import LinearProbeWrapper
from .lora import LoRALPWrapper, LoRAZSWrapper
from .full_ft import FullFTWrapper, FullFTZSWrapper

# Aliases matching paper terminology (text-guided adaptation)
LoRATextWrapper = LoRAZSWrapper
FullFTTextWrapper = FullFTZSWrapper
