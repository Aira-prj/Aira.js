# Aira.js-Beta
![Aira Snowy Owl](aira.png)
### Aira is a WebGPU-based artificial intelligence library built on a GPT-style architecture, written completely from scratch.

By running on the GPU, it delivers performance optimizations for tensor operations and training loops.

## Tests
**- Intel® HD Graphics 630**:

 **- 221 lines dataset, 128 hidden, 64 embedSize, 3 epochs, learning rate 0.003**
    ![inteltest](inteltest.png)
**- Nvidia GeForce GTX 1050 Mobile**:

   **- 221 lines dataset, 128 hidden, 64 embedSize, 3 epochs, learning rate 0.003**
   ![1050test](1050test.png)

## Features

MLP Layer: Multilayer perceptron with forward and backward propagation

MHA (Multi-Head Attention): Transformer-based attention mechanism

LayerNorm: Normalization for training stability

Tensor Operations: GPU-accelerated matrix and vector multiplications

NLP Modules: BPE tokenization with encode/decode support

Training Loop: Training an MLP model on data with loss tracking

Sampling: Top-K and Top-P sampling, repeat penalty, and temperature control

## Hardware Requirements

**Minimum(for testing and experimentation):**
- CPU: Modern x64 CPU
- GPU: Intel HD Graphics 630 or equivalent
- RAM: 8 GB
- VRAM: ~1.5 GB (shared or dedicated)

**Recommended:**
- GPU: NVIDIA GTX 1650 / RTX 2080 or better
- Ram: 16-32 gb
- VRAM: 8–16 GB

**Contact me**:aira.project@proton.me

## Please give a star to this project — it’s by far the biggest project I’ve worked on so far.

## Quick Start
```html
<script type="module" src="src/Aira.js">

(async () => {
  const device = await ensureGPU();
  const embedSize = 128
  const hiddenSize = 512
  const trainedModel = await StartModel("http://127.0.0.1:5500/datasets/en_Aira_big_chat_ds.txt", 500, embedSize, hiddenSize, 5, 0.003);
  const usrprompt = "Hello!";
  const mha = new MHA(device,embedSize,3)
  const layernorm = new LayerNorm(embedSize)
  const response = await generateReply(
    mha,
    trainedModel.mlp, 
    layernorm, 
    trainedModel.nlp, 
    trainedModel.embed, 
    usrprompt
  );
  
  console.log("Aira:", response);
})();
  
</script>
```

