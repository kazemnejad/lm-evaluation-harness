import transformers
import torch
from lm_eval.models.gpt2 import HFLM

import huggingface_hub
import accelerate




class OPT(HFLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super(HFLM, self).__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            revision=revision,
            subfolder=subfolder,
        )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(
            self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)
        ):
            assert self.tokenizer.encode(
                "hello\n\nhello", add_special_tokens=False
            ) == [42891, 50118, 50118, 42891], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

        self.pretrained = pretrained

        self._load_opt_model()

    def _load_opt_model(self):
        weights_path = huggingface_hub.snapshot_download(self.pretrained)

        # If the folder contains a checkpoint that isn't sharded, it needs to point to the state dict directly
        # otherwise point to the directory containing the shard
        import os

        files = os.listdir(weights_path)
        weights_path = (
            os.path.join(weights_path, "pytorch_model.bin")
            if "pytorch_model.bin" in files
            else weights_path
        )
        print("OPT weights path", weights_path)

        config = transformers.AutoConfig.from_pretrained(self.pretrained)

        # Initializes an empty shell with the model. This is instant and does not take any RAM.
        with accelerate.init_empty_weights():
            model = transformers.AutoModelForCausalLM.from_config(config)
        # Initialize the model under the previous context manager breaks the tied weights.
        model.tie_weights()

        # Infer device map automatically
        device_map = accelerate.infer_auto_device_map(
            model.model, no_split_module_classes=["OPTDecoderLayer"], dtype="float16"
        )

        embed_tokens_device = device_map["decoder.embed_tokens"]
        num_gpus = torch.cuda.device_count()
        num_layers = config.num_hidden_layers

        # Evenly distribute layers between gpus
        if num_gpus > 0:
            bs = ((num_layers - 1) // num_gpus) + 1
            for k, i in enumerate(range(0, num_layers - 1, bs)):
                for j in range(i, min(i + bs, num_layers - 1)):
                    device_map[f"decoder.layers.{j}"] = k

        # Last layer needs to be on the same device as the token embedding
        # because the vocabulary projection layer has its weight tied to the token embedding
        device_map[f"decoder.layers.{num_layers-1}"] = embed_tokens_device

        print(f"Device map for loading {self.pretrained}")
        print(device_map)

        accelerate.load_checkpoint_and_dispatch(
            model.model,
            weights_path,
            device_map=device_map,
            dtype="float16",
        )
        model.tie_weights()

        self.gpt2 = model
        self.gpt2.eval()


class OPTWithPhaseShift(OPT):
    def __init__(self, phase_shift=0, **kwargs):
        super().__init__(**kwargs)
        self.phase_shift = int(phase_shift)

    def _model_call(self, inps: torch.Tensor):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            if self.phase_shift != 0:
                batch_size, seq_len = inps.shape
                position_ids = torch.arange(
                    0, seq_len, dtype=torch.long, device=inps.device
                )
                position_ids = (
                    position_ids.unsqueeze(0).view(-1, seq_len).repeat(batch_size, 1)
                )
                position_ids += self.phase_shift
                position_ids = position_ids.to(inps.device)
            else:
                position_ids = None

            return self.gpt2(input_ids=inps)[0][:, :, :50257]
