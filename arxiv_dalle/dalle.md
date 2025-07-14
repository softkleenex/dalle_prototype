::: {.ltx_page_main}
::: {.ltx_page_content}
# Zero-Shot Text-to-Image Generation {#zero-shot-text-to-image-generation .ltx_title .ltx_title_document}

::: {.ltx_authors}
[ [Aditya Ramesh ]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Mikhail Pavlov
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Gabriel Goh ]{.ltx_personname}]{.ltx_creator
.ltx_role_author} [  ]{.ltx_author_before}[ [Scott Gray
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Chelsea Voss ]{.ltx_personname}]{.ltx_creator
.ltx_role_author} [  ]{.ltx_author_before}[ [Alec Radford
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
[  ]{.ltx_author_before}[ [Mark Chen ]{.ltx_personname}]{.ltx_creator
.ltx_role_author} [  ]{.ltx_author_before}[ [Ilya Sutskever
]{.ltx_personname}]{.ltx_creator .ltx_role_author}
:::

::: {.ltx_abstract}
###### Abstract {#abstract .ltx_title .ltx_title_abstract}

Text-to-image generation has traditionally focused on finding better
modeling assumptions for training on a fixed dataset. These assumptions
might involve complex architectures, auxiliary losses, or side
information such as object part labels or segmentation masks supplied
during training. We describe a simple approach for this task based on a
transformer that autoregressively models the text and image tokens as a
single stream of data. With sufficient data and scale, our approach is
competitive with previous domain-specific models when evaluated in a
zero-shot fashion.
:::

::: {.ltx_keywords}
Machine Learning, ICML
:::

::: {#p2 .ltx_para}
\
:::

::: {#S1 .section .ltx_section}
## [1 ]{.ltx_tag .ltx_tag_section}Introduction {#introduction .ltx_title .ltx_title_section}

::: {#S1.p1 .ltx_para}
Modern machine learning approaches to text to image synthesis started
with the work of [mansimov2015generating]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}, who showed that the DRAW [gregor2015draw]{.ltx_ref
.ltx_missing_citation .ltx_ref_self} generative model, when extended to
condition on image captions, could also generate novel visual scenes.
[reed2016generative]{.ltx_ref .ltx_missing_citation .ltx_ref_self} later
demonstrated that using a generative adversarial network
([goodfellow2014generative]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), rather than a recurrent variational auto-encoder,
improved image fidelity. [reed2016generative]{.ltx_ref
.ltx_missing_citation .ltx_ref_self} showed that this system could not
only generate objects with recognizable properties, but also could
[zero-shot]{.ltx_text .ltx_font_italic} generalize to held-out
categories.
:::

::: {#S1.p2 .ltx_para}
Over the next few years, progress continued using a combination of
methods. These include improving the generative model architecture with
modifications like multi-scale generators ([zhang2017stackgan]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}; [zhang2018stackgan++]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}), integrating attention and
auxiliary losses ([xu2018attngan]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), and leveraging additional sources of conditioning
information beyond just text ([reed2016learning]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}; [li2019object]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}; [koh2021text]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}).
:::

![[Figure 1: ]{.ltx_tag .ltx_tag_figure}Comparison of original images
(top) and reconstructions from the discrete VAE (bottom). The encoder
downsamples the spatial resolution by a factor of 8. While details
(e.g., the texture of the cat's fur, the writing on the storefront, and
the thin lines in the illustration) are sometimes lost or distorted, the
main features of the image are still typically recognizable. We use a
large vocabulary size of 8192 to mitigate the loss of
information.](dvae_rec.png){#S1.F1.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="598" height="396"}

::: {#S1.p3 .ltx_para}
Separately, [nguyen2017plug]{.ltx_ref .ltx_missing_citation
.ltx_ref_self} propose an energy-based framework for conditional image
generation that obtained a large improvement in sample quality relative
to contemporary methods. Their approach can incorporate pretrained
discriminative models, and they show that it is capable of performing
text-to-image generation when applied to a captioning model pretrained
on MS-COCO. More recently, [cho2020x]{.ltx_ref .ltx_missing_citation
.ltx_ref_self} also propose a method that involves optimizing the input
to a pretrained cross-modal masked language model. While significant
increases in visual fidelity have occurred as a result of the work since
[mansimov2015generating]{.ltx_ref .ltx_missing_citation .ltx_ref_self},
samples can still suffer from severe artifacts such as object
distortion, illogical object placement, or unnatural blending of
foreground and background elements.
:::

::: {#S1.p4 .ltx_para}
Recent advances fueled by large-scale generative models suggest a
possible route for further improvements. Specifically, when compute,
model size, and data are scaled carefully, autoregressive transformers
([vaswani2017attention]{.ltx_ref .ltx_missing_citation .ltx_ref_self})
have achieved impressive results in several domains such as text
([radford2019language]{.ltx_ref .ltx_missing_citation .ltx_ref_self}),
images ([chen2020generative]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), and audio ([dhariwal2020jukebox]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}).
:::

![[(a) ]{.ltx_tag .ltx_tag_figure}a tapir made of accordion. a tapir
with the texture of an accordion.](tapir_0.png){#S1.F2.sf1.g1
.ltx_graphics .ltx_img_square width="52" height="52"}

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(b) ]{.ltx_tag .ltx_tag_figure}an illustration of a baby hedgehog in
a christmas sweater walking a dog](hedgehog_1.png){#S1.F2.sf2.g1
.ltx_graphics .ltx_img_square width="52" height="52"}
:::

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(c) ]{.ltx_tag .ltx_tag_figure}a neon sign that reads "backprop". a
neon sign that reads "backprop". backprop neon
sign](10.png){#S1.F2.sf3.g1 .ltx_graphics .ltx_img_square width="52"
height="52"}
:::

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(d) ]{.ltx_tag .ltx_tag_figure}the exact same cat on the top as a
sketch on the bottom](30.png){#S1.F2.sf4.g1 .ltx_graphics
.ltx_img_square width="52" height="52"}
:::

[Figure 2: ]{.ltx_tag .ltx_tag_figure}With varying degrees of
reliability, our model appears to be able to combine distinct concepts
in plausible ways, create anthropomorphized versions of animals, render
text, and perform some types of image-to-image translation.

::: {#S1.p5 .ltx_para}
By comparison, text-to-image generation has typically been evaluated on
relatively small datasets such as MS-COCO and CUB-200
([welinder2010caltech]{.ltx_ref .ltx_missing_citation .ltx_ref_self}).
Could dataset size and model size be the limiting factor of current
approaches? In this work, we demonstrate that training a 12-billion
parameter autoregressive transformer on 250 million image-text pairs
collected from the internet results in a flexible, high fidelity
generative model of images controllable through natural language.
:::

::: {#S1.p6 .ltx_para}
The resulting system achieves high quality image generation on the
popular MS-COCO dataset [zero-shot]{.ltx_text .ltx_font_italic}, without
using any of the training labels. It is preferred over prior work
trained on the dataset by human evaluators 90% of the time. We also find
that it is able to perform complex tasks such as image-to-image
translation at a rudimentary level. This previously required custom
approaches ([isola2017image]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), rather emerging as a capability of a single, large
generative model.
:::

![[Figure 3: ]{.ltx_tag .ltx_tag_figure}Comparison of samples from our
model to those from prior approaches on captions from MS-COCO. Each of
our model samples is the best of 512 as ranked by the contrastive model.
We do not use any manual cherrypicking with the selection of either the
captions or the samples from any of the
models.](coco_cmp_v2.jpg){#S1.F3.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="598" height="399"}
:::

::: {#S2 .section .ltx_section}
## [2 ]{.ltx_tag .ltx_tag_section}Method {#method .ltx_title .ltx_title_section}

::: {#S2.p1 .ltx_para}
Our goal is to train a transformer ([vaswani2017attention]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) to autoregressively model the text
and image tokens as a single stream of data. However, using pixels
directly as image tokens would require an inordinate amount of memory
for high-resolution images. Likelihood objectives tend to prioritize
modeling short-range dependencies between pixels
([salimans2017pixelcnn++]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), so much of the modeling capacity would be spent
capturing high-frequency details instead of the low-frequency structure
that makes objects visually recognizable to us.
:::

::: {#S2.p2 .ltx_para}
We address these issues by using a two-stage training procedure, similar
to ([oord2017neural]{.ltx_ref .ltx_missing_citation .ltx_ref_self};
[razavi2019generating]{.ltx_ref .ltx_missing_citation .ltx_ref_self}):

-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I1.i1}
    ::: {#S2.I1.i1.p1 .ltx_para}
    [Stage 1.]{.ltx_text .ltx_font_bold} We train a discrete variational
    autoencoder (dVAE)[^1^[[^1^ [1]{.ltx_tag .ltx_tag_note}
    <https://github.com/openai/DALL-E>]{.ltx_note_content}]{.ltx_note_outer}]{#footnote1
    .ltx_note .ltx_role_footnote} to compress each $256 \times 256$ RGB
    image into a $32 \times 32$ grid of image tokens, each element of
    which can assume $8192$ possible values. This reduces the context
    size of the transformer by a factor of $192$ without a large
    degradation in visual quality (see Figure [[1]{.ltx_text
    .ltx_ref_tag}](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I1.i2}
    ::: {#S2.I1.i2.p1 .ltx_para}
    [Stage 2.]{.ltx_text .ltx_font_bold} We concatenate up to 256
    BPE-encoded text tokens with the ${32 \times 32} = 1024$ image
    tokens, and train an autoregressive transformer to model the joint
    distribution over the text and image tokens.
    :::
:::

::: {#S2.p3 .ltx_para}
The overall procedure can be viewed as maximizing the evidence lower
bound (ELB) ([kingma2013auto]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}; [rezende2014stochastic]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) on the joint likelihood of the model distribution over
images $x$, captions $y$, and the tokens $z$ for the encoded RGB image.
We model this distribution using the factorization
${p_{\theta,\psi}{(x,y,z)}} = {p_{\theta}{(\left. x \middle| {y,z} \right.)}p_{\psi}{(y,z)}}$,
which yields the lower bound

  -- ------------------------------------------------------------------------ -- ----------------------------------------------------
     $$\begin{matrix}                                                            [(1)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
     {\ln p_{\theta,\psi}{(x,y)} \geqslant \underset{\begin{matrix}              
      \\                                                                         
     {z \sim {q_{\phi}{({z|x})}}} \\                                             
     \end{matrix}}{\mathbb{E}}\left( \ln p_{\theta}{(x|y,z)} - \right.} \\       
     {\left. \beta D_{KL}{(q_{\phi}{(y,z|x)},p_{\psi}{(y,z)})} \right),} \\      
     \end{matrix}$$                                                              
  -- ------------------------------------------------------------------------ -- ----------------------------------------------------

where:

-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I2.i1}
    ::: {#S2.I2.i1.p1 .ltx_para}
    $q_{\phi}$ denotes the distribution over the $32 \times 32$ image
    tokens generated by the dVAE encoder given the RGB
    image $x$[^2^[[^2^ [2]{.ltx_tag .ltx_tag_note} We assume that $y$ is
    conditionally independent of $x$
    given $z$.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote2
    .ltx_note .ltx_role_footnote};
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I2.i2}
    ::: {#S2.I2.i2.p1 .ltx_para}
    $p_{\theta}$ denotes the distribution over the RGB images generated
    by the dVAE decoder given the image tokens; and
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I2.i3}
    ::: {#S2.I2.i3.p1 .ltx_para}
    $p_{\psi}$ denotes the joint distribution over the text and image
    tokens modeled by the transformer.
    :::

Note that the bound only holds for $\beta = 1$, while in practice we
find it helpful to use larger values ([higgins2016beta]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}). The following subsections
describe both stages in further detail.[^3^[[^3^ [3]{.ltx_tag
.ltx_tag_note} In preliminary experiments on
ImageNet ([deng2009imagenet]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), we attempted to maximize the ELB with respect
to $\phi$, $\theta$, and $\psi$ jointly, but were unable to improve on
two-stage training.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote3
.ltx_note .ltx_role_footnote}
:::

::: {#S2.SS1 .section .ltx_subsection}
### [2.1 ]{.ltx_tag .ltx_tag_subsection}Stage One: Learning the Visual Codebook {#stage-one-learning-the-visual-codebook .ltx_title .ltx_title_subsection}

::: {#S2.SS1.p1 .ltx_para}
In the first stage of training, we maximize the ELB with respect
to $\phi$ and $\theta$, which corresponds to training a dVAE on the
images alone. We set the initial prior $p_{\psi}$ to the uniform
categorical distribution over the $K = 8192$ codebook vectors, and
$q_{\phi}$ to be categorical distributions parameterized by the $8192$
logits at the same spatial position in the $32 \times 32$ grid output by
the encoder.
:::

::: {#S2.SS1.p2 .ltx_para}
The ELB now becomes difficult to optimize: as $q_{\psi}$ is a discrete
distribution, and we cannot use the reparameterization gradient to
maximize it. [oord2017neural]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}; [razavi2019generating]{.ltx_ref .ltx_missing_citation
.ltx_ref_self} address this using an online cluster assignment procedure
coupled with the straight-through
estimator ([bengio2013estimating]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}). We instead use the gumbel-softmax
relaxation ([jang2016categorical]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}; [maddison2016concrete]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), replacing the expectation over $q_{\phi}$ with one over
$q_{\phi}^{\tau}$, where the relaxation becomes tight as the
temperature $\tau\rightarrow 0$. The likelihood for $p_{\theta}$ is
evaluated using the log-laplace distribution (see
Appendix [[A.3]{.ltx_text
.ltx_ref_tag}](#A1.SS3 "A.3 The Logit-Laplace Distribution ‣ Appendix A Details for Discrete VAE ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
for a derivation).
:::

::: {#S2.SS1.p3 .ltx_para}
The relaxed ELB is maximized using Adam ([kingma2014adam]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) with exponentially weighted
iterate averaging. Appendix [[A.2]{.ltx_text
.ltx_ref_tag}](#A1.SS2 "A.2 Training ‣ Appendix A Details for Discrete VAE ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
gives a complete description of the hyperparameters, but we found the
following to be especially important for stable training:

-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I3.i1}
    ::: {#S2.I3.i1.p1 .ltx_para}
    Specific annealing schedules for the relaxation temperature and step
    size. We found that annealing $\tau$ to $1/16$ was sufficient to
    close the gap between the relaxed validation ELB and the true
    validation ELB with $q_{\phi}$ intsead of $q_{\phi}^{\tau}$.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I3.i2}
    ::: {#S2.I3.i2.p1 .ltx_para}
    The use of $1 \times 1$ convolutions at the end of the encoder and
    the beginning of the decoder. We found that reducing the receptive
    field size for the convolutions around the relaxation led to it
    generalizing better to the true ELB.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I3.i3}
    ::: {#S2.I3.i3.p1 .ltx_para}
    Multiplication of the outgoing activations from the encoder and
    decoder resblocks by a small constant, to ensure stable training at
    initialization.
    :::

We also found that increasing the KL weight to $\beta = 6.6$ promotes
better codebook usage and ultimately leads to a *smaller* reconstruction
error at the end of training.[^4^[[^4^ [4]{.ltx_tag .ltx_tag_note} This
is contrary to the usual tradeoff between the two terms. We speculate
that for smaller values of $\beta$, the noise from the relaxation causes
the optimizer to reduce codebook usage toward the beginning of training,
resulting in worse ELB at
convergence.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote4 .ltx_note
.ltx_role_footnote}
:::
:::

::: {#S2.SS2 .section .ltx_subsection}
### [2.2 ]{.ltx_tag .ltx_tag_subsection}Stage Two: Learning the Prior {#stage-two-learning-the-prior .ltx_title .ltx_title_subsection}

::: {#S2.SS2.p1 .ltx_para}
In the second stage, we fix $\phi$ and $\theta$, and learn the prior
distribution over the text and image tokens by maximizing the ELB with
respect to $\psi$. Here, $p_{\psi}$ is represented by a 12-billion
parameter sparse transformer ([child2019generating]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}).
:::

::: {#S2.SS2.p2 .ltx_para}
Given a text-image pair, we BPE-encode ([sennrich2015neural]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) the lowercased caption using at
most 256 tokens[^5^[[^5^ [5]{.ltx_tag .ltx_tag_note} During training, we
apply 10% BPE dropout ([provilkov2019bpe]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), whose use is common in the neural machine translation
literature.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote5 .ltx_note
.ltx_role_footnote} with vocabulary size $16384$, and encode the image
using ${32 \times 32} = 1024$ tokens with vocabulary size $8192$. The
image tokens are obtained using argmax sampling from the dVAE encoder
logits, without adding any gumbel noise.[^6^[[^6^ [6]{.ltx_tag
.ltx_tag_note} Strictly speaking, Equation [[1]{.ltx_text
.ltx_ref_tag}](#S2.E1 "In 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
requires us to sample from the categorical distribution specified by the
dVAE encoder logits, rather than taking the argmax. In preliminary
experiments on ImageNet, we found that this was a useful regularizer in
the overparameterized regime, and allows the transformer to be trained
using soft targets for the cross-entropy loss. We decided against this
here since the model in consideration is in the underparameterized
regime.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote6 .ltx_note
.ltx_role_footnote} Finally, the text and image tokens are concatenated
and modeled autoregressively as a single stream of data.
:::

::: {#S2.SS2.p3 .ltx_para}
The transformer is a decoder-only model in which each image token can
attend to all text tokens in any one of its 64 self-attention layers.
The full architecture is described in Appendix [[B.1]{.ltx_text
.ltx_ref_tag}](#A2.SS1 "B.1 Architecture ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
There are three different kinds of self-attention masks used in the
model. The part of the attention masks corresponding to the text-to-text
attention is the standard causal mask, and the part for the
image-to-image attention uses either a row, column, or convolutional
attention mask.[^7^[[^7^ [7]{.ltx_tag .ltx_tag_note} We found using a
single attention operation for all three interactions -- "text attends
to text", "image attends to text", and "image attends to image" -- to
perform better than using separate attention operations that are
independently
normalized.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote7 .ltx_note
.ltx_role_footnote}
:::

::: {#S2.SS2.p4 .ltx_para}
We limit the length of a text caption to 256 tokens, though it is not
totally clear what to do for the "padding" positions in between the last
text token and the start-of-image token. One option is to set the logits
for these tokens to $- \infty$ in the self-attention operations.
Instead, we opt to learn a special padding token separately for each of
the 256 text positions. This token is used only when no text token is
available. In preliminary experiments on Conceptual
Captions ([sharma2018conceptual]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), we found that this resulted in higher validation loss,
but better performance on out-of-distribution captions.
:::

::: {#S2.SS2.p5 .ltx_para}
We normalize the cross-entropy losses for the text and image tokens by
the total number of each kind in a batch of data. Since we are primarily
interested in image modeling, we multiply the cross-entropy loss for the
text by $1/8$ and the cross-entropy loss for the image by $7/8$. The
objective is optimized using Adam with exponentially weighted iterate
averaging; Appendix [[B.2]{.ltx_text
.ltx_ref_tag}](#A2.SS2 "B.2 Training ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
describes the training procedure in more detail. We reserved
about $606000$ images for validation, and found no signs of overfitting
at convergence.
:::

![[Figure 4: ]{.ltx_tag .ltx_tag_figure}Illustration of per-resblock
gradient scaling for a transformer resblock. The solid line indicates
the sequence of operations for forward propagation, and the dashed line
the sequence of operations for backpropagation. We scale the incoming
gradient for each resblock by its gradient scale, and unscale the
outgoing gradient before it is added to the sum of the gradients from
the successive resblocks. The activations and gradients along the
identity path are stored in 32-bit precision. The "filter" operation
sets all Inf and NaN values in the activation gradient to zero. Without
this, a nonfinite event in the current resblock would cause the gradient
scales for all preceding resblocks to unnecessarily drop, thereby
resulting in underflow.](per_resblock_scaling.png){#S2.F4.g1
.ltx_graphics .ltx_centering .ltx_img_landscape width="598"
height="449"}

![[Figure 5: ]{.ltx_tag .ltx_tag_figure}Communication patterns used for
distributed training. Each parameter array in the model is sharded among
the eight GPUs on each machine. During forward propagation, we prefetch
the parameter shards for the next resblock (using all-gather) while
computing the activations for the current resblock. To conserve memory,
the parameter shards from the other GPUs are immediately discarded.
Similarly, during backpropagation, we prefetch the parameter shards for
the previous resblock while computing the activations and gradients for
the current resblock. After all GPUs have computed the gradient with
respect to an all-gathered parameter, the reduce-scatter operation
leaves each GPU with only one slice -- i.e., the gradient for its
parameter shard, averaged over the eight GPUs.](dist_comm.png){#S2.F5.g1
.ltx_graphics .ltx_centering .ltx_img_square width="299" height="299"}
:::

::: {#S2.SS3 .section .ltx_subsection}
### [2.3 ]{.ltx_tag .ltx_tag_subsection}Data Collection {#data-collection .ltx_title .ltx_title_subsection}

::: {#S2.SS3.p1 .ltx_para}
Our preliminary experiments for models up to $1.2$ billion parameters
were carried out on Conceptual Captions, a dataset of 3.3 million
text-image pairs that was developed as an extension to
MS-COCO ([lin2014microsoft]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}).
:::

::: {#S2.SS3.p2 .ltx_para}
To scale up to $12$-billion parameters, we created a dataset of a
similar scale to JFT-300M ([sun2017revisiting]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) by collecting 250 million
text-images pairs from the internet. This dataset does not include
MS-COCO, but does include Conceptual Captions and a filtered subset of
YFCC100M ([thomee2016yfcc100m]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}). As MS-COCO was created from the latter, our training
data includes a fraction of the MS-COCO validation images (but none of
the captions). We control for this in the quantitative results presented
in Section [[3]{.ltx_text
.ltx_ref_tag}](#S3 "3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
and find that it has no appreciable bearing on the results. We provide
further details about the data collection process in
Appendix [[C]{.ltx_text
.ltx_ref_tag}](#A3 "Appendix C Details for Data Collection ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
:::
:::

::: {#S2.SS4 .section .ltx_subsection}
### [2.4 ]{.ltx_tag .ltx_tag_subsection}Mixed-Precision Training {#mixed-precision-training .ltx_title .ltx_title_subsection}

::: {#S2.SS4.p1 .ltx_para}
To save GPU memory and increase throughput, most parameters, Adam
moments, and activations are stored in 16-bit precision. We also use
activation checkpointing and recompute the activations within the
resblocks during the backward pass. Getting the model to train in 16-bit
precision past one billion parameters, without diverging, was the most
challenging part of this project.
:::

::: {#S2.SS4.p2 .ltx_para}
We believe the root cause of this instability to be underflow in the
16-bit gradients. Appendix [[D]{.ltx_text
.ltx_ref_tag}](#A4 "Appendix D Guidelines for Mixed-Precision Training ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
presents a set of guidelines we developed to avoid underflow when
training large-scale generative models. Here, we describe one of these
guidelines: per-resblock gradient scaling.
:::

::: {#S2.SS4.p3 .ltx_para}
Similar to prior work ([liu2020understanding]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}), we found that the norms of the
activation gradients from the resblocks decrease monotonically as we
move from the earlier resblocks to the later ones.[^8^[[^8^ [8]{.ltx_tag
.ltx_tag_note} It is possible that better initialization
schemes ([liu2020understanding]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) might be able to avoid this, but we did not have success
with alternative schemes in our
experiments.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote8 .ltx_note
.ltx_role_footnote} As the model is made deeper and wider, the true
exponents of the activation gradients for later resblocks can fall below
the minimum exponent of the 16-bit format. Consequently, they get
rounded to zero, a phenomenon called *underflow*. We found that
eliminating underflow allowed for stable training to convergence.
:::

::: {#S2.SS4.p4 .ltx_para}
Standard loss scaling ([micikevicius2017mixed]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) is able to avoid underflow when
the range spanned by the smallest and largest activation gradients (in
absolute value) fits within the exponent range of the 16-bit format. On
NVIDIA V100 GPUs, this exponent range is specified by five bits. While
this is sufficient for training vanilla language models of the same
size, we found the range to be too small for the text-to-image model.
:::

::: {#S2.SS4.p5 .ltx_para}
Our fix, which is shown in Figure [[4]{.ltx_text
.ltx_ref_tag}](#S2.F4 "Figure 4 ‣ 2.2 Stage Two: Learning the Prior ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
involves using a separate "gradient scale" for each resblock in the
model. This can be seen as a practical alternative to a more general
framework for mixed-precision training called
Flexpoint ([koster2017flexpoint]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), with the advantage that specialized GPU kernels are not
required. We found that [sun2020ultra]{.ltx_ref .ltx_missing_citation
.ltx_ref_self} had independently developed similar procedure for
training convolutional networks in 4-bit precision.
:::
:::

::: {#S2.SS5 .section .ltx_subsection}
### [2.5 ]{.ltx_tag .ltx_tag_subsection}Distributed Optimization {#distributed-optimization .ltx_title .ltx_title_subsection}

  [Effective Parameter Count]{.ltx_text style="font-size:70%;"}                                                      [Compression Rank]{.ltx_text style="font-size:70%;"}   [Compression Rate]{.ltx_text style="font-size:70%;"}
  ------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------ ------------------------------------------------------
  $2.8 \cdot 10^{9}$[ (]{.ltx_text style="font-size:70%;"}$d_{model} = 1920$[)]{.ltx_text style="font-size:70%;"}    [512]{.ltx_text style="font-size:70%;"}                $\approx {83\%}$
  $5.6 \cdot 10^{9}$[ (]{.ltx_text style="font-size:70%;"}$d_{model} = 2688$[)]{.ltx_text style="font-size:70%;"}    [640]{.ltx_text style="font-size:70%;"}                $\approx {85\%}$
  $12.0 \cdot 10^{9}$[ (]{.ltx_text style="font-size:70%;"}$d_{model} = 3968$[)]{.ltx_text style="font-size:70%;"}   [896]{.ltx_text style="font-size:70%;"}                $\approx {86\%}$

[Table 1: ]{.ltx_tag .ltx_tag_table}We show the relationship between
model size and the minimum compression rank for the gradients (up to a
multiple of 128) necessary to avoid a gap in the training loss during
the first $10\%$ of training. These results suggest that in our setting,
we can achieve a compression rate of about $85\%$, independent of model
size.

![[Figure 6: ]{.ltx_tag .ltx_tag_figure}Effect of increasing the number
of images for the contrastive reranking procedure on MS-COCO
captions.](coco_reranking.jpg){#S2.F6.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="598" height="312"}

::: {#S2.SS5.p1 .ltx_para}
Our 12-billion parameter model consumes about 24 GB of memory when
stored in 16-bit precision, which exceeds the memory of a 16 GB NVIDIA
V100 GPU. We address this using parameter
sharding ([rajbhandari2019zero]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}). As shown in Figure [[5]{.ltx_text
.ltx_ref_tag}](#S2.F5 "Figure 5 ‣ 2.2 Stage Two: Learning the Prior ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
parameter sharding allows us to almost completely hide the latency of
the intra-machine communication by overlapping it with compute-intensive
operations.
:::

::: {#S2.SS5.p2 .ltx_para}
On the cluster used to train the model, the bandwidth between machines
is much lower than the bandwidth among GPUs on the same machine. This
makes the cost of the operation used to average the gradient among the
machines (all-reduce) the main bottleneck during training. We were able
to drastically reduce this cost by compressing the gradients using
PowerSGD ([vogels2019powersgd]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}).
:::

::: {#S2.SS5.p3 .ltx_para}
In our implementation, each GPU in a machine computes the low-rank
factors for its parameter shard gradients independently of its
neighboring GPUs.[^9^[[^9^ [9]{.ltx_tag .ltx_tag_note} There is still
intra-machine communication for other operations; what we mean is that
the low-rank factors across the shards, when concatenated, are not
regarded as collectively approximating the gradient for the full
parameter matrix.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote9
.ltx_note .ltx_role_footnote} Once the low-rank factors are computed,
each machine sets its error buffer to the residual between the
uncompressed gradient averaged over its eight GPUs (obtained from
reduce-scatter), and the decompressed gradient obtained from the
low-rank factors.
:::

::: {#S2.SS5.p4 .ltx_para}
PowerSGD replaces the large communication operation for an uncompressed
parameter gradient with two, much smaller communication operations for
its low-rank factors. For a given compression rank $r$ and transformer
activation size $d_{model}$, the compression rate is given
by $1 - {{5r}/{({8d_{\text{model}}})}}$ (see Appendix [[E.1]{.ltx_text
.ltx_ref_tag}](#A5.SS1 "E.1 Bandwidth Analysis ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
Table [[1]{.ltx_text
.ltx_ref_tag}](#S2.T1 "Table 1 ‣ 2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
shows that we can achieve a compression rate of about $85\%$,
independent of model size.
:::

::: {#S2.SS5.p5 .ltx_para}
In Appendix [[E.2]{.ltx_text
.ltx_ref_tag}](#A5.SS2 "E.2 Implementation Details ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
we describe various details that were necessary to get PowerSGD to
perform well at scale. These include:

-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I4.i1}
    ::: {#S2.I4.i1.p1 .ltx_para}
    Saving memory by accumulating the gradient into the error buffers
    during backpropagation, rather than allocating separate buffers.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I4.i2}
    ::: {#S2.I4.i2.p1 .ltx_para}
    Minimizing instances in which we zero out the error buffers (e.g.,
    due to nonfinite values encountered during mixed-precision
    backpropagation, or when resuming training from a checkpoint).
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I4.i3}
    ::: {#S2.I4.i3.p1 .ltx_para}
    Improving numerical stability by using Householder orthogonalization
    instead of Gram-Schmidt, together with the addition of a small
    multiple of the identity matrix to the input.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#S2.I4.i4}
    ::: {#S2.I4.i4.p1 .ltx_para}
    Avoiding underflow by using a custom 16-bit floating point format
    for the error buffers, their low-rank factors, and the all-reduce
    communication operations involving them.
    :::

We also found the warm-start procedure for the $Q$ matrix described in
[vogels2019powersgd]{.ltx_ref .ltx_missing_citation .ltx_ref_self} to be
unnecessary: we were able to get equivalent results by fixing $Q$ to a
random gaussian matrix at the start of training, and never updating
it.[^10^[[^10^ [10]{.ltx_tag .ltx_tag_note} We verified that the error
in reconstructing the true gradient is higher when $Q$ is fixed as
opposed to being updated using warm-starting, so it is interesting that
this does not affect the loss. By contrast, resampling $Q$ at every
update causes a large performance
hit.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote10 .ltx_note
.ltx_role_footnote}
:::
:::

::: {#S2.SS6 .section .ltx_subsection}
### [2.6 ]{.ltx_tag .ltx_tag_subsection}Sample Generation {#sample-generation .ltx_title .ltx_title_subsection}

::: {#S2.SS6.p1 .ltx_para}
Similar to [razavi2019generating]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}, we rerank the samples drawn from the transformer using a
pretrained contrastive model ([radford2021learning]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}). Given a caption and a candidate
image, the contrastive model assigns a score based on how well the image
matches the caption. Figure [[6]{.ltx_text
.ltx_ref_tag}](#S2.F6 "Figure 6 ‣ 2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
shows the effect of increasing the number of samples $N$ from which we
select the top $k$ images. This process can be seen as a kind of
language-guided search ([andreas2017learning]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}), and is also similar to the
auxiliary text-image matching loss proposed by [xu2018attngan]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}. Unless otherwise stated, all
samples used for both qualitative and quantitative results are obtained
without temperature reduction (i.e., using $t = 1$) (except for
Figure [[2]{.ltx_text
.ltx_ref_tag}](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref})
and use reranking with $N = 512$.
:::
:::
:::

::: {#S3 .section .ltx_section}
## [3 ]{.ltx_tag .ltx_tag_section}Experiments {#experiments .ltx_title .ltx_title_section}

![[Figure 7: ]{.ltx_tag .ltx_tag_figure}Human evaluation of our model
(evaluated zero-shot without temperature reduction) vs prior
work (DF-GAN) on captions from MS-COCO. In a best-of-five vote, our
model's sample was chosen as the most realistic 90.0% of the time, and
was chosen as the image best matching a shared caption 93.3% of the
time.](assets_v2_final_graph.png){#S3.F7.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="598" height="449"}

::: {#S3.SS1 .section .ltx_subsection}
### [3.1 ]{.ltx_tag .ltx_tag_subsection}Quantitative Results {#quantitative-results .ltx_title .ltx_title_subsection}

::: {#S3.SS1.p1 .ltx_para}
We evaluate our model zero-shot by comparing it to three prior
approaches: AttnGAN ([xu2018attngan]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), DM-GAN ([zhu2019dm]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), and DF-GAN ([tao2020df]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}), the last of which reports the best Inception
Score ([salimans2016improved]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) and Fréchet Inception
Distance ([heusel2017gans]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) on MS-COCO. Figure [[3]{.ltx_text
.ltx_ref_tag}](#S1.F3 "Figure 3 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
qualitatively compares samples from our model to those from prior work.
:::

::: {#S3.SS1.p2 .ltx_para}
We also conduct a human evaluation similar to the one used
in [koh2021text]{.ltx_ref .ltx_missing_citation .ltx_ref_self} to
compare our approach to DF-GAN, the results of which are shown in
Figure [[7]{.ltx_text
.ltx_ref_tag}](#S3.F7 "Figure 7 ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
Given a caption, the sample from our model receives the majority vote
for better matching the caption 93% of the time. It also receives the
majority vote for being more realistic 90% of the time.
:::

::: {#S3.SS1.p3 .ltx_para}
Figure [[9]{.ltx_text
.ltx_ref_tag}](#S3.F9 "Figure 9 ‣ 3.1 Quantitative Results ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(a)
shows that our model also obtains an FID score on MS-COCO within
2 points of the best prior approach, despite having never been trained
on the captions. Our training data incorporates a filtered subset of
YFCC100M, and we found that it includes about $21\%$ of the images in
the MS-COCO validation set from a de-duplication procedure described in
the next section. To isolate this effect, we compute the FID statistics
for the validation set both with these images (solid lines) and without
them (dashed lines), finding no significant change in the results.
:::

::: {#S3.SS1.p4 .ltx_para}
Training the transformer on the tokens from the dVAE encoder allows us
to allocate its modeling capacity to the low-frequency information that
makes images visually recognizable to us. However, it also disadvantages
the model, since the heavy compression renders it unable to produce
high-frequency details. To test the effect of this on the quantitative
evaluations, we compute the FID and IS in Figure [[9]{.ltx_text
.ltx_ref_tag}](#S3.F9 "Figure 9 ‣ 3.1 Quantitative Results ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(a)
after applying a Gaussian filter with varying radius to both the
validation images and samples from the models. Our approach achieves the
best FID by a margin of about 6 points with a slight blur of radius 1.
The gap between our approach and others tends to widen as the blur
radius is increased. We also obtain the highest IS when the blur radius
is greater than or equal to two.
:::

![[Figure 8: ]{.ltx_tag .ltx_tag_figure}Zero-shot samples from our model
on the CUB dataset.](cub_samples.jpg){#S3.F8.g1 .ltx_graphics
.ltx_centering .ltx_img_landscape width="598" height="450"}

![[(a) ]{.ltx_tag .ltx_tag_figure}FID and IS on MS-COCO as a function of
blur radius.](coco_fid.png){#S3.F9.sf1.g1 .ltx_graphics
.ltx_img_landscape width="186" height="118"}

::: {.ltx_flex_cell .ltx_flex_size_2}
![[(b) ]{.ltx_tag .ltx_tag_figure}FID and IS on CUB as a function of
blur radius.](cub_fid.png){#S3.F9.sf2.g1 .ltx_graphics
.ltx_img_landscape width="186" height="118"}
:::

::: {.ltx_flex_break}
:::

::: {.ltx_flex_cell .ltx_flex_size_1}
![[(c) ]{.ltx_tag .ltx_tag_figure}FID and IS on MS-COCO as a function of
the sample size used for
reranking.](coco_fid_clip_sample_size.png){#S3.F9.sf3.g1 .ltx_graphics
.ltx_img_landscape width="186" height="118"}
:::

[Figure 9: ]{.ltx_tag .ltx_tag_figure}Quantitative results on MS-COCO
and CUB. Solid lines represent FID computed against the original
validation sets, and dashed lines represent FID computed against
validation sets with overlapping images removed (see
Section [[3.2]{.ltx_text
.ltx_ref_tag}](#S3.SS2 "3.2 Data Overlap Analysis ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
For MS-COCO, we evaluate all models on a subset of $30000$ captions
sampled from the validation set. For CUB, we evaluate all models on all
of the unique captions in the test set. We compute the FID and IS using
the DM-GAN code, which is available at
<https://github.com/MinfengZhu/DM-GAN>.

::: {#S3.SS1.p5 .ltx_para}
Our model fares significantly worse on the CUB dataset, for which there
is a nearly 40-point gap in FID between our model and the leading prior
approach (Figure [[9]{.ltx_text
.ltx_ref_tag}](#S3.F9 "Figure 9 ‣ 3.1 Quantitative Results ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(b)).
We found an $12\%$ overlap rate for this dataset, and again observed no
significant difference in the results after removing these images. We
speculate that our zero-shot approach is less likely to compare
favorably on specialized distributions such as CUB. We believe that
fine-tuning is a promising direction for improvement, and leave this
investigation to future work. Samples from our model for captions in
this dataset are shown in Figure [[8]{.ltx_text
.ltx_ref_tag}](#S3.F8 "Figure 8 ‣ 3.1 Quantitative Results ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
:::

::: {#S3.SS1.p6 .ltx_para}
Finally, Figure [[9]{.ltx_text
.ltx_ref_tag}](#S3.F9 "Figure 9 ‣ 3.1 Quantitative Results ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(c)
shows clear improvements in FID and IS for MS-COCO as the sample size
used for reranking with the contrastive model is increased. This trend
continues up to a sample size of 32, after which we observe diminishing
returns.
:::
:::

::: {#S3.SS2 .section .ltx_subsection}
### [3.2 ]{.ltx_tag .ltx_tag_subsection}Data Overlap Analysis {#data-overlap-analysis .ltx_title .ltx_title_subsection}

::: {#S3.SS2.p1 .ltx_para}
We used the deduplication procedure described
in [radford2021learning]{.ltx_ref .ltx_missing_citation .ltx_ref_self}
to determine which images to remove. For each validation image, we find
the closest image in the training data using a contrastive model
specifically trained for this task. We then sort the images in
descending order by closeness to their nearest matches in the training
data. After inspecting the results by hand, we determine the images to
remove by manually selecting a conservative threshold designed to
minimize the false negative rate.
:::
:::

::: {#S3.SS3 .section .ltx_subsection}
### [3.3 ]{.ltx_tag .ltx_tag_subsection}Qualitative Findings {#qualitative-findings .ltx_title .ltx_title_subsection}

::: {#S3.SS3.p1 .ltx_para}
We found that our model has the ability to generalize in ways that we
did not originally anticipate. When given the caption "a tapir made of
accordion..." (Figure [[2a]{.ltx_text
.ltx_ref_tag}](#S1.F2.sf1 "In Figure 2 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}),
the model appears to draw a tapir with an accordion for a body, or an
accordion whose keyboard or bass are in the shape of a tapir's trunk or
legs. This suggests that it has developed a rudimentary ability to
compose unusual concepts at high levels of abstraction.
:::

::: {#S3.SS3.p2 .ltx_para}
Our model also appears to be capable of combinatorial generalization,
such as when rendering text (Figure [[2b]{.ltx_text
.ltx_ref_tag}](#S1.F2.sf2 "In Figure 2 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref})
or when probed on sentences like "an illustration of a baby hedgehog in
a christmas sweater walking a dog" (Figure [[2c]{.ltx_text
.ltx_ref_tag}](#S1.F2.sf3 "In Figure 2 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
Prompts like the latter require the model to perform variable
binding ([smolensky1990tensor]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}; [greff2020binding]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) -- it is the hedgehog that is in the christmas sweater,
not the dog. We note, however, that the model performs inconsistently on
the task, sometimes drawing both animals with christmas sweaters, or
drawing a hedgehog walking a smaller hedgehog.
:::

::: {#S3.SS3.p3 .ltx_para}
To a limited degree of reliability, we also find our model to be capable
of zero-shot image-to-image translation controllable by natural language
(Figure [[2d]{.ltx_text
.ltx_ref_tag}](#S1.F2.sf4 "In Figure 2 ‣ 1 Introduction ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
When the model is given the caption "the exact same cat on the top as a
sketch at the bottom" and the top $15 \times 32$ part of the image token
grid for a photo of a cat, it is able to draw a sketch of a similar
looking cat on the bottom.
:::

::: {#S3.SS3.p4 .ltx_para}
This works with several other kinds of transformations, including image
operations (e.g., changing the color of the image, converting it to
grayscale, or flipping it upside-down) and style transfer (e.g., drawing
the cat on a greeting card, a postage stamp, or a cell phone case). Some
transformations, such as those that involve only changing the color of
the animal, suggest that the model is capable of performing a
rudimentary kind of object segmentation. We provide additional examples
of zero-shot image-to-image translation in Section [[G]{.ltx_text
.ltx_ref_tag}](#A7 "Appendix G Zero-Shot Image-to-Image Translation ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
:::
:::
:::

::: {#S4 .section .ltx_section}
## [4 ]{.ltx_tag .ltx_tag_section}Conclusion {#conclusion .ltx_title .ltx_title_section}

::: {#S4.p1 .ltx_para}
We investigate a simple approach for text-to-image generation based on
an autoregressive transformer, when it is executed at scale. We find
that scale can lead to improved generalization, both in terms of
zero-shot performance relative to previous domain-specific approaches,
and in terms of the range of capabilities that emerge from a single
generative model. Our findings suggest that improving generalization as
a function of scale may be a useful driver for progress on this task.
:::
:::

::: {#Sx1 .section .ltx_section}
## Acknowledgements {#acknowledgements .ltx_title .ltx_title_section}

::: {#Sx1.p1 .ltx_para}
We would like to thank Matthew Knight for reviewing the code release for
this work, and Rewon Child, John Schulman, Heewoo Jun, and Prafulla
Dhariwal for helpful early feedback on the paper. We would also like to
thank Jong Wook Kim for writing the PyTorch package for the contrastive
model described in [radford2019language]{.ltx_ref .ltx_missing_citation
.ltx_ref_self} that we used to rerank the samples from our model.
:::
:::

::: {#bib .section .ltx_bibliography}
## References {#references .ltx_title .ltx_title_bibliography}
:::

::: {.ltx_pagination .ltx_role_newpage}
:::

::: {#A1 .section .ltx_appendix}
## [Appendix A ]{.ltx_tag .ltx_tag_appendix}Details for Discrete VAE {#appendix-a-details-for-discrete-vae .ltx_title .ltx_title_appendix}

::: {#A1.SS1 .section .ltx_subsection}
### [A.1 ]{.ltx_tag .ltx_tag_subsection}Architecture {#a.1-architecture .ltx_title .ltx_title_subsection}

::: {#A1.SS1.p1 .ltx_para}
The dVAE encoder and decoder are
convolutional ([lecun1998gradient]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) ResNets ([he2016identity]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) with bottleneck-style resblocks. The models primarily
use $3 \times 3$ convolutions, with $1 \times 1$ convolutions along skip
connections in which the number of feature maps changes between the
input and output of a resblock. The first convolution of the encoder
is $7 \times 7$, and the last convolution of the encoder (which produces
the $32 \times 32 \times 8192$ output used as the logits for the
categorical distributions for the image tokens) is $1 \times 1$. Both
the first and last convolutions of the decoder are $1 \times 1$. The
encoder uses max-pooling (which we found to yield better ELB than
average-pooling) to downsample the feature maps, and the decoder uses
nearest-neighbor upsampling. The precise details for the architectures
are given in the files [dvae/encoder.py]{.ltx_text .ltx_font_typewriter}
and [dvae/decoder.py]{.ltx_text .ltx_font_typewriter} of the code
release.
:::
:::

::: {#A1.SS2 .section .ltx_subsection}
### [A.2 ]{.ltx_tag .ltx_tag_subsection}Training {#a.2-training .ltx_title .ltx_title_subsection}

::: {.ltx_listing .ltx_lst_language_Python .ltx_lstlisting .ltx_listing}
::: {.ltx_listing_data}
[⬇](data:text/plain;base64,ZGVmIHByZXByb2Nlc3NfaW1hZ2UoaW1nLCB0YXJnZXRfcmVzKToKICAgIGgsIHcgID0gdGYuc2hhcGUoaW1nKVswXSwgdGYuc2hhcGUoaW1nKVsxXQogICAgc19taW4gPSB0Zi5taW5pbXVtKGgsIHcpCiAgICBpbWcgICA9IHRmLmltYWdlLnJhbmRvbV9jcm9wKGltZywgMiAqIFtzX21pbl0gKyBbM10pCgogICAgdF9taW4gPSB0Zi5taW5pbXVtKHNfbWluLCByb3VuZCg5IC8gOCAqIHRhcmdldF9yZXMpKQogICAgdF9tYXggPSB0Zi5taW5pbXVtKHNfbWluLCByb3VuZCgxMiAvIDggKiB0YXJnZXRfcmVzKSkKICAgIHQgICAgID0gdGYucmFuZG9tLnVuaWZvcm0oW10sIHRfbWluLCB0X21heCArIDEsIGR0eXBlPXRmLmludDMyKQogICAgaW1nICAgPSB0Zi5pbWFnZS5yZXNpemVfaW1hZ2VzKGltZywgW3QsIHRdLCBtZXRob2Q9dGYuaW1hZ2UuUmVzaXplTWV0aG9kLkFSRUEsCiAgICAgICAgICAgICAgICBhbGlnbl9jb3JuZXJzPVRydWUpCiAgICBpbWcgICA9IHRmLmNhc3QodGYucmludCh0Zi5jbGlwX2J5X3ZhbHVlKGltZywgMCwgMjU1KSksIHRmLnVpbnQ4KQogICAgaW1nICAgPSB0Zi5pbWFnZS5yYW5kb21fY3JvcChpbWcsIDIgKiBbdGFyZ2V0X3Jlc10gKyBbY2hhbm5lbF9jb3VudF0pCiAgICByZXR1cm4gdGYuaW1hZ2UucmFuZG9tX2ZsaXBfbGVmdF9yaWdodChpbWcp)
:::

::: {#lstnumberx1 .ltx_listingline}
[def]{.ltx_text .ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[preprocess_image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[target_res]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[):]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx2 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[w]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[=]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[shape]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)\[0\],]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[shape]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)\[1\]]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx3 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[s_min]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[minimum]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[h]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx4 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[random_crop]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[2]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\[]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[\]]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\[3\])]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx5 .ltx_listingline}
:::

::: {#lstnumberx6 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t_min]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[minimum]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[round]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(9]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[/]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[8]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[target_res]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[))]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx7 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t_max]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[minimum]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[round]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(12]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[/]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[8]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[target_res]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[))]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx8 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[random]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uniform]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(\[\],]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t_min]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[t_max]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[1,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[dtype]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[int32]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx9 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[resize_images]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\[]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[t]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[\],]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[method]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[=]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[image]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ResizeMethod]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[AREA]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx10 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[align_corners]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[True]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx11 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[cast]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[rint]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[clip_by_value]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[0,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[255)),]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uint8]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx12 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[random_crop]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[2]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\[]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[target_res]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[\]]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\[]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[channel_count]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[\])]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx13 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[return]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter
style="font-size:80%;"}[random_flip_left_right]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::
:::

[Listing 1: ]{.ltx_tag
.ltx_tag_float}TensorFlow ([abadi2016tensorflow]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) image preprocessing code for
training dVAE. We use [target_res = 256]{.ltx_text .ltx_font_typewriter}
and [channel_count = 3]{.ltx_text .ltx_font_typewriter}.

::: {#A1.SS2.p1 .ltx_para}
The dVAE is trained on the same dataset as the transformer, using the
data augmentation code given in Listing [[1]{.ltx_text
.ltx_ref_tag}](#LST1 "Listing 1 ‣ A.2 Training ‣ Appendix A Details for Discrete VAE ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
Several quantities are decayed during training, all of which use a
cosine schedule:

1.  [[1.]{.ltx_tag .ltx_tag_item}]{#A1.I1.i1}
    ::: {#A1.I1.i1.p1 .ltx_para}
    The KL weight $\beta$ is increased from $0$ to $6.6$ over the
    first $5000$ updates. [bowman2015generating]{.ltx_ref
    .ltx_missing_citation .ltx_ref_self} use a similar schedule based on
    the sigmoid function.
    :::
2.  [[2.]{.ltx_tag .ltx_tag_item}]{#A1.I1.i2}
    ::: {#A1.I1.i2.p1 .ltx_para}
    The relaxation temperature $\tau$ is annealed from $1$ to $1/16$
    over the first $150000$ updates. Using a linear annealing schedule
    for this typically led to divergence.
    :::
3.  [[3.]{.ltx_tag .ltx_tag_item}]{#A1.I1.i3}
    ::: {#A1.I1.i3.p1 .ltx_para}
    The step size is annealed from $1 \cdot 10^{- 4}$
    to $1.25 \cdot 10^{- 6}$ over $1200000$ updates.
    :::

The decay schedules for the relaxation temperature and the step size are
especially important for stability and successful optimization.
:::

::: {#A1.SS2.p2 .ltx_para}
We update the parameters using AdamW ([loshchilov2017decoupled]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) with $\beta_{1} = 0.9$,
$\beta_{2} = 0.999$, $\epsilon = 10^{- 8}$, and weight decay
multiplier $10^{- 4}$. We use exponentially weighted iterate averaging
for the parameters with decay coefficient $0.999$. The reconstruction
term in the ELB is a joint distribution over
the $256 \times 256 \times 3$ values for the image pixels, and the KL
term is a joint distribution over the $32 \times 32$ positions in the
spatial grid output by the encoder. We divide the overall loss
by $256 \times 256 \times 3$, so that the weight of the KL term
becomes $\beta/192$, where $\beta$ is the KL weight. The model is
trained in mixed-precision using standard (i.e., global) loss scaling
on $64$ 16 GB NVIDIA V100 GPUs, with a per-GPU batch size of $8$,
resulting in a total batch size of 512. It is trained for a total
of $3000000$ updates.
:::
:::

::: {#A1.SS3 .section .ltx_subsection}
### [A.3 ]{.ltx_tag .ltx_tag_subsection}The Logit-Laplace Distribution {#a.3-the-logit-laplace-distribution .ltx_title .ltx_title_subsection}

::: {#A1.SS3.p1 .ltx_para}
The $\ell_{1}$ and $\ell_{2}$ reconstruction objectives are commonly
used when training VAEs. These objectives correspond to using Laplace
and Gaussian distributions
for ${\ln p_{\theta}}{(\left. x \middle| {y,z} \right.)}$ in
Equation [[1]{.ltx_text
.ltx_ref_tag}](#S2.E1 "In 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
respectively. There is a strange mismatch in this modeling choice: pixel
values lie within a bounded interval, but both of these distributions
are supported by the entire real line. Hence, some amount of likelihood
will be placed outside the admissible range of pixel values.
:::

::: {#A1.SS3.p2 .ltx_para}
We present a variant of the Laplace distribution that is also supported
by a bounded interval. This resolves the discrepancy between the range
of the pixel values being modeled and the support of the distribution
used to model them. We consider the pdf of the random variable obtained
by applying the sigmoid function to a Laplace-distributed random
variable. This pdf is defined on $(0,1)$ and is given by

  -- ----------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------
     $${{f{(\left. x \middle| {\mu,b} \right.)}} = {\frac{1}{2bx{({1 - x})}}{\exp\left( {- \frac{|{{{logit}{(x)}} - \mu}|}{b}} \right)}}};$$      [(2)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ----------------------------------------------------------------------------------------------------------------------------------------- -- ----------------------------------------------------

we call it the *logit-Laplace distribution.* We use the logarithm of the
RHS of Equation [[2]{.ltx_text
.ltx_ref_tag}](#A1.E2 "In A.3 The Logit-Laplace Distribution ‣ Appendix A Details for Discrete VAE ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
as the reconstruction term for the training objective of the dVAE.
:::

::: {#A1.SS3.p3 .ltx_para}
The decoder of the dVAE produces six feature maps representing the
sufficient statistics of the logit-Laplace distribution for the RGB
channels of the image being reconstructed. The first three feature maps
represent the $\mu$ parameter for the RGB channels, and the last three
represent $\ln b$. Before feeding an image into the dVAE encoder, we
transform its values
using $\varphi:{{\lbrack 0,255\rbrack}\rightarrow{(\epsilon,{1 - \epsilon})}}$,
which is given by

  -- ---------------------------------------------------------------------- -- ----------------------------------------------------
     $${\varphi:{x\mapsto{{\frac{1 - {2\epsilon}}{255}x} + \epsilon}}}.$$      [(3)]{.ltx_tag .ltx_tag_equation .ltx_align_right}
  -- ---------------------------------------------------------------------- -- ----------------------------------------------------

This restricts the range of the pixel values to be modeled by the dVAE
decoder to $(\epsilon,{1 - \epsilon})$, which avoids numerical problems
arising from the $x{({1 - x})}$ in Equation [[2]{.ltx_text
.ltx_ref_tag}](#A1.E2 "In A.3 The Logit-Laplace Distribution ‣ Appendix A Details for Discrete VAE ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
We use $\epsilon = 0.1$. To reconstruct an image for manual inspection
or computing metrics, we ignore $\ln b$ and
compute $\hat{x} = {\varphi^{- 1}{({{sigmoid}{(\mu)}})}}$, where $\mu$
is given by the first three feature maps output by the dVAE
decoder.[^11^[[^11^ [11]{.ltx_tag .ltx_tag_note} See
[notebooks/usage.ipynb]{.ltx_text .ltx_font_typewriter} of the code
release for an
example.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote11 .ltx_note
.ltx_role_footnote}
:::
:::
:::

::: {#A2 .section .ltx_appendix}
## [Appendix B ]{.ltx_tag .ltx_tag_appendix}Details for Transformer {#appendix-b-details-for-transformer .ltx_title .ltx_title_appendix}

::: {#A2.SS1 .section .ltx_subsection}
### [B.1 ]{.ltx_tag .ltx_tag_subsection}Architecture {#b.1-architecture .ltx_title .ltx_title_subsection}

![[Figure 10: ]{.ltx_tag .ltx_tag_figure}Illustration of the embedding
scheme for a hypothetical version of our transformer with a maximum text
length of 6 tokens. Each box denotes a vector of
size $d_{model} = 3968$. In this illustration, the caption has a length
of 4 tokens, so 2 padding tokens are used (as described in
Section [[2.2]{.ltx_text
.ltx_ref_tag}](#S2.SS2 "2.2 Stage Two: Learning the Prior ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
Each image vocabulary embedding is summed with a row and column
embedding.](xf_embds.png){#A2.F10.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="598" height="157"}

![[(a) ]{.ltx_tag .ltx_tag_figure}Row attention
mask.](attn_row.png){#A2.F11.sf1.g1 .ltx_graphics .ltx_img_square
width="144" height="144"}

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(b) ]{.ltx_tag .ltx_tag_figure}Column attention
mask.](attn_col.png){#A2.F11.sf2.g1 .ltx_graphics .ltx_img_square
width="144" height="144"}
:::

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(c) ]{.ltx_tag .ltx_tag_figure}Column attention mask with transposed
image states.](attn_col_t.png){#A2.F11.sf3.g1 .ltx_graphics
.ltx_img_square width="144" height="144"}
:::

::: {.ltx_flex_cell .ltx_flex_size_4}
![[(d) ]{.ltx_tag .ltx_tag_figure}Convolutional attention
mask.](attn_conv.png){#A2.F11.sf4.g1 .ltx_graphics .ltx_img_square
width="144" height="144"}
:::

[Figure 11: ]{.ltx_tag .ltx_tag_figure}Illustration of the three types
of attention masks for a hypothetical version of our transformer with a
maximum text length of 6 tokens and image length of 16 tokens (i.e.,
corresponding to a $4 \times 4$ grid). Mask (a) corresponds to row
attention in which each image token attends to the previous 5 image
tokens in raster order. The extent is chosen to be 5, so that the last
token being attended to is the one in the same column of the previous
row. To obtain better GPU utilization, we transpose the row and column
dimensions of the image states when applying column attention, so that
we can use mask (c) instead of mask (b). Mask (d) corresponds to a
causal convolutional attention pattern with wraparound behavior (similar
to the row attention) and a $3 \times 3$ kernel. Our model uses a mask
corresponding to an $11 \times 11$ kernel.

::: {#A2.SS1.p1 .ltx_para}
Our model is a decoder-only sparse transformer of the same kind
described in [child2019generating]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}, with broadcasted row and column embeddings for the part
of the context for the image tokens. A complete description of the
embedding scheme used in our model is shown in Figure [[10]{.ltx_text
.ltx_ref_tag}](#A2.F10 "Figure 10 ‣ B.1 Architecture ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
We use 64 attention layers, each of which uses 62 attention heads with a
per-head state size of 64.
:::

::: {#A2.SS1.p2 .ltx_para}
The model uses three kinds of sparse attention masks, which we show in
Figure [[11]{.ltx_text
.ltx_ref_tag}](#A2.F11 "Figure 11 ‣ B.1 Architecture ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
The convolutional attention mask (Figure [[11]{.ltx_text
.ltx_ref_tag}](#A2.F11 "Figure 11 ‣ B.1 Architecture ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(d))
is only used in the last self-attention layer. Otherwise, given the
index $i$ of a self-attention layer
(with $i \in {\lbrack 1,63\rbrack}$), we use the column attention mask
(Figure [[11]{.ltx_text
.ltx_ref_tag}](#A2.F11 "Figure 11 ‣ B.1 Architecture ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}(c))
if ${{i - 2}\operatorname{mod}4} = 0$, and row attention otherwise.
E.g., the first four self-attention layers use "row, column, row, row",
respectively. With the exception of the convolutional attention mask,
which we found to provide a small boost in performance over the row and
dense causal attention masks when used in the final self-attention
layer, this is the same configuration used
in [child2019generating]{.ltx_ref .ltx_missing_citation .ltx_ref_self}.
:::
:::

::: {#A2.SS2 .section .ltx_subsection}
### [B.2 ]{.ltx_tag .ltx_tag_subsection}Training {#b.2-training .ltx_title .ltx_title_subsection}

::: {.ltx_listing .ltx_lst_language_Python .ltx_lstlisting .ltx_listing}
::: {.ltx_listing_data}
[⬇](data:text/plain;base64,ZGVmIHByZXByb2Nlc3NfaW1hZ2UoaW1nLCB0YXJnZXRfcmVzKToKICAgIGgsIHcgID0gdGYuc2hhcGUoaW1nKVswXSwgdGYuc2hhcGUoaW1nKVsxXQogICAgc19taW4gPSB0Zi5taW5pbXVtKGgsIHcpCgogICAgb2ZmX2ggPSB0Zi5yYW5kb20udW5pZm9ybShbXSwgMyAqIChoIC0gc19taW4pIC8vIDgsCiAgICAgICAgdGYubWF4aW11bSgzICogKGggLSBzX21pbikgLy8gOCArIDEsIDUgKiAoaCAtIHNfbWluKSAvLyA4KSwKICAgICAgICBkdHlwZT10Zi5pbnQzMikKICAgIG9mZl93ID0gdGYucmFuZG9tLnVuaWZvcm0oW10sIDMgKiAodyAtIHNfbWluKSAvLyA4LAogICAgICAgIHRmLm1heGltdW0oMyAqICh3IC0gc19taW4pIC8vIDggKyAxLCA1ICogKHcgLSBzX21pbikgLy8gOCksCiAgICAgICAgZHR5cGU9dGYuaW50MzIpCgogICAgIyBSYW5kb20gZnVsbCBzcXVhcmUgY3JvcC4KICAgIGltZyAgID0gdGYuaW1hZ2UuY3JvcF90b19ib3VuZGluZ19ib3goaW1nLCBvZmZfaCwgb2ZmX3csIHNfbWluLCBzX21pbikKICAgIHRfbWF4ID0gdGYubWluaW11bShzX21pbiwgcm91bmQoOSAvIDggKiB0YXJnZXRfcmVzKSkKICAgIHQgICAgID0gdGYucmFuZG9tLnVuaWZvcm0oW10sIHRhcmdldF9yZXMsIHRfbWF4ICsgMSwgZHR5cGU9dGYuaW50MzIpCiAgICBpbWcgICA9IHRmLmltYWdlLnJlc2l6ZV9pbWFnZXMoaW1nLCBbdCwgdF0sIG1ldGhvZD10Zi5pbWFnZS5SZXNpemVNZXRob2QuQVJFQSwKICAgICAgICAgICAgICAgIGFsaWduX2Nvcm5lcnM9VHJ1ZSkKICAgIGltZyAgID0gdGYuY2FzdCh0Zi5yaW50KHRmLmNsaXBfYnlfdmFsdWUoaW1nLCAwLCAyNTUpKSwgdGYudWludDgpCgogICAgIyBXZSBkb24ndCB1c2UgaGZsaXAgYXVnIHNpbmNlIHRoZSBpbWFnZSBtYXkgY29udGFpbiB0ZXh0LgogICAgcmV0dXJuIHRmLmltYWdlLnJhbmRvbV9jcm9wKGltZywgMiAqIFt0YXJnZXRfcmVzXSArIFtjaGFubmVsX2NvdW50XSk=)
:::

::: {#lstnumberx14 .ltx_listingline}
[def]{.ltx_text .ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[preprocess_image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[target_res]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[):]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx15 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[w]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[=]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[shape]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)\[0\],]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[shape]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)\[1\]]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx16 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[s_min]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[minimum]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[h]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx17 .ltx_listingline}
:::

::: {#lstnumberx18 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[off_h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[random]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uniform]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(\[\],]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[3]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx19 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[maximum]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(3]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[1,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[5]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[h]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8),]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx20 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[dtype]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[int32]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx21 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[off_w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[random]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uniform]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(\[\],]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[3]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx22 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[maximum]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(3]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[1,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[5]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\*]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[-]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[//]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[8),]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx23 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[dtype]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[int32]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx24 .ltx_listingline}
:::

::: {#lstnumberx25 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\#]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[Random]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[full]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[square]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[crop]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx26 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter
style="font-size:80%;"}[crop_to_bounding_box]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[off_h]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[off_w]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[s_min]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx27 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t_max]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[minimum]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[s_min]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[round]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(9]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[/]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[8]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[target_res]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[))]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx28 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[random]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uniform]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(\[\],]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[target_res]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[t_max]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[1,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[dtype]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[int32]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[)]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::

::: {#lstnumberx29 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[resize_images]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\[]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[t]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[t]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[\],]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[method]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[=]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[image]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ResizeMethod]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[AREA]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx30 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[align_corners]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[True]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx31 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[=]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[cast]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[rint]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[clip_by_value]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[(]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[img]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[0,]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[255)),]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[tf]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[uint8]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[)]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx32 .ltx_listingline}
:::

::: {#lstnumberx33 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\#]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[We]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[don]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[']{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[t]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[use]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[hflip]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[aug]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[since]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[the]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[may]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter style="font-size:80%;"}[
]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[contain]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[text]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}
:::

::: {#lstnumberx34 .ltx_listingline}
[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[return]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[tf]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[.]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[image]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[.]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[random_crop]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[(]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[img]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[,]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[2]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\*]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space
.ltx_font_typewriter style="font-size:80%;"}[\[]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}[target_res]{.ltx_text
.ltx_lst_identifier .ltx_font_typewriter
style="font-size:80%;"}[\]]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[+]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[ ]{.ltx_text .ltx_lst_space .ltx_font_typewriter
style="font-size:80%;"}[\[]{.ltx_text .ltx_font_typewriter
style="font-size:80%;"}[channel_count]{.ltx_text .ltx_lst_identifier
.ltx_font_typewriter style="font-size:80%;"}[\])]{.ltx_text
.ltx_font_typewriter style="font-size:80%;"}
:::
:::

[Listing 2: ]{.ltx_tag
.ltx_tag_float}TensorFlow ([abadi2016tensorflow]{.ltx_ref
.ltx_missing_citation .ltx_ref_self}) image preprocessing code for
training the transformer. We use [target_res = 256]{.ltx_text
.ltx_font_typewriter} and [channel_count = 3]{.ltx_text
.ltx_font_typewriter}.

::: {#A2.SS2.p1 .ltx_para}
When training the transformer, we apply data augmentation to the images
before encoding them using the dVAE encoder. We use slightly different
augmentations from the ones used to train the dVAE; the code used for
this is given in Listing [[2]{.ltx_text
.ltx_ref_tag}](#LST2 "Listing 2 ‣ B.2 Training ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
We also apply 10% BPE dropout when BPE-encoding the captions for
training. The model is trained using per-resblock scaling (see
Section [[2.4]{.ltx_text
.ltx_ref_tag}](#S2.SS4 "2.4 Mixed-Precision Training ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref})
and gradient compression (see Section [[2.5]{.ltx_text
.ltx_ref_tag}](#S2.SS5 "2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref})
with total compression rank 896 (so that each GPU uses a compression
rank of 112 for its parameter shards). As shown in Table [[1]{.ltx_text
.ltx_ref_tag}](#S2.T1 "Table 1 ‣ 2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
this results in a compression rate of about 86%, which we analyze in
Section [[E.1]{.ltx_text
.ltx_ref_tag}](#A5.SS1 "E.1 Bandwidth Analysis ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
:::

::: {#A2.SS2.p2 .ltx_para}
We update the parameters using AdamW with $\beta_{1} = 0.9$,
$\beta_{2} = 0.96$, $\epsilon = 10^{- 8}$, and weight decay
multiplier $4.5 \cdot 10^{- 2}$. We clip the decompressed gradients by
norm using a threshold of 4, prior to applying the Adam update. Gradient
clipping is only triggered during the warm-up phase at the start of
training. To conserve memory, most Adam moments (see
Section [[D]{.ltx_text
.ltx_ref_tag}](#A4 "Appendix D Guidelines for Mixed-Precision Training ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
for details) are stored in 16-bit formats, with a 1-6-9 format for the
running mean (i.e., 1 bit for the sign, 6 bits for the exponent, and
9 bits for the significand), and a 0-6-10 format for the running
variance. We clip the estimate for running variance by value to 5 before
it is used to update the parameters or moments. Finally, we apply
exponentially weighted iterate averaging by asynchronously copying the
model parameters from the GPU to the CPU once every 25 updates, using a
decay coefficient of 0.99.
:::

::: {#A2.SS2.p3 .ltx_para}
We trained the model using 1024, 16 GB NVIDIA V100 GPUs and a total
batch size of $1024$, for a total of $430000$ updates. At the start of
training, we use a linear schedule to ramp up the step size
to $4.5 \cdot 10^{- 4}$ over $5000$ updates, and halved the step size
each time the training loss appeared to plateau. We did this a total of
five times, ending training with a final step size that was 32 times
smaller than the initial one. We reserved about $606000$ images for
validation, and did not observe overfitting at any point during
training.
:::
:::
:::

::: {#A3 .section .ltx_appendix}
## [Appendix C ]{.ltx_tag .ltx_tag_appendix}Details for Data Collection {#appendix-c-details-for-data-collection .ltx_title .ltx_title_appendix}

::: {#A3.p1 .ltx_para}
In order to train the 12-billion parameter transformer, we created a
dataset of a similar scale to JFT-300M by collecting 250 million
text-image pairs from the internet. As described in
Section [[2.3]{.ltx_text
.ltx_ref_tag}](#S2.SS3 "2.3 Data Collection ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
this dataset incorporates Conceptual Captions, the text-image pairs from
Wikipedia, and a filtered subset of YFCC100M. We use a subset of the
text, image, and joint text and image filters described in
[sharma2018conceptual]{.ltx_ref .ltx_missing_citation .ltx_ref_self} to
construct this dataset. These filters include discarding instances whose
captions are too short, are classified as non-English by the Python
package [cld3]{.ltx_text .ltx_font_typewriter}, or that consist
primarily of boilerplate phrases such as "photographed on
[\<date\>]{.ltx_text .ltx_font_typewriter}", where [\<date\>]{.ltx_text
.ltx_font_typewriter} matches various formats for dates that we found in
the data. We also discard instances whose images have aspect ratios not
in $\lbrack{1/2},2\rbrack$. If we were to use to very tall or wide
images, then the square crops used during training would likely exclude
objects mentioned in the caption.
:::
:::

::: {#A4 .section .ltx_appendix}
## [Appendix D ]{.ltx_tag .ltx_tag_appendix}Guidelines for Mixed-Precision Training {#appendix-d-guidelines-for-mixed-precision-training .ltx_title .ltx_title_appendix}

![[Figure 12: ]{.ltx_tag .ltx_tag_figure}Plot of per-resblock gradient
scales for a 2.8-billion parameter text-to-image transformer trained
without gradient compression. The $x$-axis is parameter updates, and the
$y$-axis is the base-2 logarithm of the gradient scale. Darkest violet
corresponds to the first resblock, and brightest yellow corresponds to
the last (of which there are 128 total). The gradient scale for the
second MLP resblock hovers at around $2^{24}$, while the others stay
within a 4-bit range. The extent of this range increases as the model is
made larger.](grad_scale_plot.png){#A4.F12.g1 .ltx_graphics
.ltx_centering .ltx_img_portrait width="240" height="321"}

::: {#A4.p1 .ltx_para}
The most challenging part of this project was getting the model to train
in 16-bit precision past one billion parameters. We were able to do this
after detecting for underflow in various parts of training, and revising
the code to eliminate it. We developed a set of guidelines as a result
of this process that we present here.[^12^[[^12^ [12]{.ltx_tag
.ltx_tag_note} Fewer of these guidelines may be necessary on hardware
like the TPU that has native support for the bfloat16 format, since the
larger 8-bit exponent range makes underflow less likely to
occur.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote12 .ltx_note
.ltx_role_footnote}

1.  [[1.]{.ltx_tag .ltx_tag_item}]{#A4.I1.i1}
    ::: {#A4.I1.i1.p1 .ltx_para}
    [Use per-resblock gradient scaling (Figure [[4]{.ltx_text
    .ltx_ref_tag}](#S2.F4 "Figure 4 ‣ 2.2 Stage Two: Learning the Prior ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref})
    instead of standard loss scaling.]{.ltx_text .ltx_font_bold} Our
    model uses 128 gradient scales, one for each of its resblocks. All
    of the gradient scales are initialized to $M \cdot 2^{13}$,
    where $M$ is the number of data-parallel replicas (i.e., the number
    of GPUs). In our setup, each grad scale is multiplied
    by $2^{1/1000}$ at every parameter update when there are no
    nonfinite values for any parameter gradient in that resblock.
    Otherwise, we divide the grad scale by $\sqrt{2}$ and skip the
    update. We also disallow consecutive divisions of the same grad
    scale within a window of $125$ updates. All grad scales are clamped
    to the range $\lbrack{M \cdot 2^{7}},{M \cdot 2^{24}}\rbrack$ after
    being updated. Figure [[12]{.ltx_text
    .ltx_ref_tag}](#A4.F12 "Figure 12 ‣ Appendix D Guidelines for Mixed-Precision Training ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
    shows the gradient scales in the early phase of training for a
    2.8-billion parameter model.
    :::
2.  [[2.]{.ltx_tag .ltx_tag_item}]{#A4.I1.i2}
    ::: {#A4.I1.i2.p1 .ltx_para}
    [Only use 16-bit precision where it is really necessary for
    performance.]{.ltx_text .ltx_font_bold} In particular, store all
    gains, biases, embeddings, and unembeddings in 32-bit precision,
    with 32-bit gradients (including for remote communication) and
    32-bit Adam moments. We disable gradient compression for these
    parameters (though PowerSGD would not make sense for 1D parameters
    like gains and biases). The logits for the text and image tokens are
    computed and stored in 32-bit precision. We found that storing the
    embeddings in 16-bit precision sometimes caused divergence early in
    optimization, and using 16-bit logits resulted in a small shift in
    the training curve, so we switched to use 32-bit precision out of an
    abundance of caution.
    :::
3.  [[3.]{.ltx_tag .ltx_tag_item}]{#A4.I1.i3}
    ::: {#A4.I1.i3.p1 .ltx_para}
    [Avoid underflow when dividing the gradient.]{.ltx_text
    .ltx_font_bold} For data-parallel training, we need to divide the
    gradients by the total number of data-parallel workers $M$. One way
    to do this is to divide the loss by the per-machine batch size, and
    then divide the parameter gradients by $M$ before summing them over
    the machines (using all-reduce). To save time and space, the
    gradients are usually computed and stored in 16-bit precision.
    When $M$ is large, this division could result in underflow before
    the gradients are summed. On the other hand, if we attempt to sum
    the gradients first and then divide them later, we could encounter
    overflow in the all-reduce.
    :::

    ::: {#A4.I1.i3.p2 .ltx_para}
    Our solution for this problem attempts to minimize the loss of
    information in the division prior to the all-reduce, without danger
    of overflow. To do this, we divide the loss by the overall batch
    size (which includes $M$ as a factor) rather than the per-machine
    batch size, and multiply the gradient scales by $M$ to compensate,
    as described in (1). Then, prior to the all-reduce operation, we
    divide the gradients by a constant that was tuned by hand to avoid
    both underflow and overflow. This was done by inspecting histograms
    of the exponents (i.e., base-2 logarithms) of the absolute values of
    the scalar components of the per-parameter gradients. Since the
    gradient scaling keeps the gradients close to right end of the
    exponent range of the 16-bit format, we found that the same constant
    worked well for all parameters in the model with 16-bit gradients.
    When using PowerSGD, we chose different constants for the $P$
    and $Q$ matrices.
    :::
:::
:::

::: {#A5 .section .ltx_appendix}
## [Appendix E ]{.ltx_tag .ltx_tag_appendix}Details for Distributed Optimization {#appendix-e-details-for-distributed-optimization .ltx_title .ltx_title_appendix}

::: {#A5.p1 .ltx_para}
We use PowerSGD ([vogels2019powersgd]{.ltx_ref .ltx_missing_citation
.ltx_ref_self}) to compress the gradients with respect to all parameters
except the embeddings, unembeddings, gains, and biases. In
Section [[E.1]{.ltx_text
.ltx_ref_tag}](#A5.SS1 "E.1 Bandwidth Analysis ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
we derive an expression for the reduction in the amount of data
communicated as a function of the compression rank and model size. In
Section [[E.2]{.ltx_text
.ltx_ref_tag}](#A5.SS2 "E.2 Implementation Details ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref},
we present a detailed overview of our adaptation of PowerSGD, and the
modifications we had to make in order to fix performance regressions,
some of which only manifest at billion-parameter scale.
:::

::: {#A5.SS1 .section .ltx_subsection}
### [E.1 ]{.ltx_tag .ltx_tag_subsection}Bandwidth Analysis {#e.1-bandwidth-analysis .ltx_title .ltx_title_subsection}

  [Parameter Names]{.ltx_text style="font-size:90%;"}                   [Parameter Shard Gradient Shape (No Compression)]{.ltx_text style="font-size:90%;"}   $P$[ shape]{.ltx_text style="font-size:90%;"}            $Q$[ shape]{.ltx_text style="font-size:90%;"}
  --------------------------------------------------------------------- ------------------------------------------------------------------------------------- -------------------------------------------------------- -------------------------------------------------------
  [qkv and post-attention matrices]{.ltx_text style="font-size:90%;"}   $d \times \left( d/m \right)$                                                         $d \times \left( r/m \right)$                            $\left( r/m \right) \times \left( d/m \right)$
  [First MLP matrix]{.ltx_text style="font-size:90%;"}                  $d \times \left( {4d}/m \right)$                                                      $d \times \left( r/m \right)$                            $\left( r/m \right) \times \left( {4d}/m \right)$
  [Second MLP matrix]{.ltx_text style="font-size:90%;"}                 $\left( {4d}/m \right) \times d$                                                      $\left( {4d}/m \right) \times \left( r/m \right)$        $\left( r/m \right) \times d$
  [Total size]{.ltx_text style="font-size:90%;"}                        $\left. {12d^{2}}/m \right.$                                                          $\left. \left( {{5drm} + {4dr}} \right)/m^{2} \right.$   $\left. \left( {{drm} + {8dr}} \right)/m^{2} \right.$

[Table 2: ]{.ltx_tag .ltx_tag_table}We analyze the amount of data sent
from each GPU on a given machine to GPUs on other machines, in the case
where we shard the parameters among the $m$ GPUs on each machine. Here,
$r$ denotes the rank used for compression, and $d$ the transformer
hidden size. The compression ratio is given by the sum of the last two
columns of the last row, divided by the first column of the last row.
This comes out to ${r{({m + 2})}}/{({2dm})}$, which for $m = 8$ is
${{5r}/8}d$.

::: {#A5.SS1.p1 .ltx_para}
Gradient compression uses the factorization $G \approx {PQ^{t}}$,
where $P$ and $Q$ both have rank $r$. Instead of using a single
all-reduce to transmit $G$, we use two, smaller all-reduces to transmit
both $P$ and $Q^{t}$ in succession. Hence, the compression ratio is the
sum of the sizes of the $P$ and $Q$ matrices divided by the sum of the
sizes of the $G$ matrices. We shard along axis 1 for all parameters
except for the second MLP matrix. The derivation of the compression
ratio in our setup is given in Table [[2]{.ltx_text
.ltx_ref_tag}](#A5.T2 "Table 2 ‣ E.1 Bandwidth Analysis ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
We note that the choice of shard axis changes the compression ratio for
the MLP matrices. Finally, this analysis excludes the embeddings,
unembeddings, gains, and biases, for which we do not use compression.
The total fraction of the bandwidth used by these parameters becomes
smaller as the model size is increased.
:::
:::

::: {#A5.SS2 .section .ltx_subsection}
### [E.2 ]{.ltx_tag .ltx_tag_subsection}Implementation Details {#e.2-implementation-details .ltx_title .ltx_title_subsection}

::: {#A5.SS2.p1 .ltx_para}
We describe the steps in our implementation of PowerSGD in detail, since
these details were crucial in getting it to work efficiently and
reliably at billion-parameter scale.

1.  [[1.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i1}
    ::: {#A5.I1.i1.p1 .ltx_para}
    Our training setup uses a combination of parameter sharding and
    gradient compression, as described in Section [[2.5]{.ltx_text
    .ltx_ref_tag}](#S2.SS5 "2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
    During backpropagation, while recomputing the activations and
    computing the gradients for the current resblock, we prefetch the
    parameters for the preceding resblock using all-gather. Once each
    GPU has computed the gradient with respect to a full parameter
    matrix, we compute the average of the slice of the gradient
    corresponding to the GPU's parameter shard, and discard the full
    gradient immediately to conserve memory. This average is taken over
    all of the GPUs on a machine using reduce-scatter.
    :::
2.  [[2.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i2}
    ::: {#A5.I1.i2.p1 .ltx_para}
    If there are no nonfinite values in the result of the
    reduce-scatter (which could be caused by overflow in backpropagation
    or the reduce-scatter), we divide the result by the resblock's
    gradient scale, and add it to the error buffer (i.e., the buffer
    used for error correction). Otherwise, we do nothing and proceed
    with backpropagation; a single nonfinite value in the gradient means
    that the entire update will be skipped, which happens about 5% of
    the time. The error buffer uses the same 1-6-9 format used for the
    Adam mean, which we describe in Section [[B.2]{.ltx_text
    .ltx_ref_tag}](#A2.SS2 "B.2 Training ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref};
    the larger exponent range ensures that this division does not result
    in underflow. Adding the gradients directly to the error buffers
    avoids redundantly allocating another set of buffers of size equal
    to the parameter shard gradients.
    :::
3.  [[3.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i3}
    ::: {#A5.I1.i3.p1 .ltx_para}
    Once the reduce-scatter operations for the resblock have finished,
    we schedule the operations to compute the $P$ matrices from the
    errors buffers and the $Q$ matrices, whose values are fixed at the
    start of training (see Section [[2.5]{.ltx_text
    .ltx_ref_tag}](#S2.SS5 "2.5 Distributed Optimization ‣ 2 Method ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
    Both the $P$ and $Q$ matrices are stored in 1-6-9 format and have
    their values scaled by predetermined constants, as discussed in
    Section [[D]{.ltx_text
    .ltx_ref_tag}](#A4 "Appendix D Guidelines for Mixed-Precision Training ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
    :::
4.  [[4.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i4}
    ::: {#A5.I1.i4.p1 .ltx_para}
    Once each GPU has computed the $P$ matrices for the parameter shards
    in a resblock, they are averaged with the $P$ matrices from the GPUs
    with the same ordinal on all other machines, using a single, grouped
    all-reduce operation. This all-reduce is carried out in the 1-6-9
    format, using a custom kernel. The grouping results in better
    bandwidth utilization, since it avoids scheduling many all-reduce
    calls for smaller, individual parameters, each of which carries some
    overhead. We clamp any infinities in the results of the all-reduce
    to the maximum value of the 1-6-9 format (which is slightly less
    than 16), retaining the sign. With our choice of scaling factors for
    the $P$ and $Q$ matrices, this clamping happens very rarely.
    :::
5.  [[5.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i5}
    ::: {#A5.I1.i5.p1 .ltx_para}
    Once the all-reduce operation for the $P$ matrices for a resblock
    have finished, we orthogonalize the columns of the resulting
    matrices. We use a custom Householder orthogonalization kernel
    rather than Gram-Schmidt, as we found the latter to be numerically
    unstable. We also add $\epsilon I_{m \times r}$ to $P$ in order to
    ensure that the result is not near rank-deficient,
    where $\epsilon = 10^{- 6}$. Here, $I_{m \times r}$ is a rectangular
    matrix of the same size as the $P$ matrix to which it is added; it
    contains the $r \times r$ identity matrix and has zeros elsewhere.
    The orthogonalizalied $P$ matrices are stored in 1-6-9 format, but
    without scaling.
    :::
6.  [[6.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i6}
    ::: {#A5.I1.i6.p1 .ltx_para}
    Once the $P$ matrices for a resblock have been orthogonalized, we
    schedule the operations to compute the new $Q$ matrices from the
    error buffers and the $P$ matrices.
    :::
7.  [[7.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i7}
    ::: {#A5.I1.i7.p1 .ltx_para}
    Once the new $Q$ matrices for a resblock have been computed, we
    schedule another grouped all-reduce, similar to what we did for
    the $P$ matrices. As in step (4), we clamp all infinities in the
    results of the all-reduce to the maximum value of the 1-6-9 format,
    retaining the sign. The error buffers for the resblock have now been
    decomposed into low-rank factors $P$ and $Q^{t}$.
    :::
8.  [[8.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i8}
    ::: {#A5.I1.i8.p1 .ltx_para}
    The gradients for all parameters that are not compressed are grouped
    together into a single, 32-bit precision all-reduce.
    Section [[D]{.ltx_text
    .ltx_ref_tag}](#A4 "Appendix D Guidelines for Mixed-Precision Training ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
    explains why we use 32-bit precision for these parameters and their
    gradients.
    :::
9.  [[9.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i9}
    ::: {#A5.I1.i9.p1 .ltx_para}
    Once all GPUs on a machine have finished steps (7) and (8) for every
    resblock in the model, the values of the $P$ and $Q$ matrices for
    the same parameter shard on all machines will be identical. We then
    compute the global gradient norm, which is the sum of two
    quantities: (a) the sum of the squared Frobenius norms of the $Q$
    matrices over all of the parameter shards on a machine, and (b) the
    sum of the squared norms of the gradients for the parameter shards
    that do not use compression, taken over all such parameter shards on
    a machine. We need to compute this value for gradient clipping (see
    Section [[B.2]{.ltx_text
    .ltx_ref_tag}](#A2.SS2 "B.2 Training ‣ Appendix B Details for Transformer ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}).
    :::
10. [[10.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i10}
    ::: {#A5.I1.i10.p1 .ltx_para}
    While computing the global norm, we also synchronize the information
    from step (2) about which parameter shard gradients contained
    nonfinite values after the reduce-scatter. After doing this, we have
    two pieces of information for each parameter shard: (a) whether its
    error buffer from step (2) contains nonfinite values on the current
    GPU, and (b) whether $P$ or $Q$ contains nonfinite values. We cannot
    rely on the values of the $P$ and $Q$ matrices to determine (b),
    since we clamp infinities as described in step (4). If we find that
    the gradient with respect to any parameter shard on the machine
    contains nonfinite values, then we set the global norm to infinity.
    :::
11. [[11.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i11}
    ::: {#A5.I1.i11.p1 .ltx_para}
    Once all of the all-reduces have finished and the global norm has
    been computed, we can apply the parameter updates. Like
    backpropagation, the parameter updates proceed resblock-by-resblock.
    The first step is to compute the decompressed gradients by forming
    the product $PQ^{t}$ for all parameters in a given resblock. To
    avoid overflow, these products are computed in 32-bit precision. We
    can then apply the Adam update to the parameters using the
    decompressed gradients and the global norm computed in step (9). If
    the global norm is not finite, then the update to the parameters and
    Adam moments is skipped. We note that the decompressed gradient must
    be divided by the scale of the $Q$ matrix (the $P$ matrix is stored
    without scaling after orthogonalization).
    :::
12. [[12.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i12}
    ::: {#A5.I1.i12.p1 .ltx_para}
    The second step is the update to the error buffers. First, we use
    the results from step (10) to check if the $P$ and $Q$ matrices for
    a given parameter shard contain only finite values. If this is the
    case, then we divide the decompressed gradient by the total number
    of machines, and subtract it from the current value for the error
    buffer. This sets the error buffer to the difference between the
    "local" gradient averaged over the GPUs on the machine using
    reduce-scatter, and the "remote" decompressed gradient (i.e., the
    "error"). If either $P$ or $Q$ contains nonfinite values, then we
    check if the error buffer computed in step (2) contains only finite
    values. If it does, then we preserve its value and do nothing. If it
    does not, then we set it to zero. The purpose of this tedious logic
    is to set an error buffer to zero only when we must do so, because
    it has been contaminated with nonfinite values. We found that error
    buffers getting set to zero too frequently by gradient scaling
    events leads to performance regressions.
    :::
13. [[13.]{.ltx_tag .ltx_tag_item}]{#A5.I1.i13}
    ::: {#A5.I1.i13.p1 .ltx_para}
    The parameter shards whose gradients are not compressed are updated
    separately.
    :::
:::

::: {#A5.SS2.p2 .ltx_para}
We also note the following important optimizations:

1.  [[1.]{.ltx_tag .ltx_tag_item}]{#A5.I2.i1}
    ::: {#A5.I2.i1.p1 .ltx_para}
    There are several opportunities for overlap between compute and
    communication in the above steps. For example, while we are running
    step (2) for resblock $i$, we can proceed to steps (3)--(8) for all
    resblocks $j > i$. Exploiting opportunities for overlap is necessary
    to achieve good performance.
    :::
2.  [[2.]{.ltx_tag .ltx_tag_item}]{#A5.I2.i2}
    ::: {#A5.I2.i2.p1 .ltx_para}
    We throttle specific operations that are liable to exhaust all
    available memory. For example, we only prefetch the parameters from
    the preceding resblock when the reduce-scatter operations have
    finished for the current one. Otherwise, we risk running out of
    memory by holding on to the full parameters. We also throttle the
    Adam updates, so that we do not decompress all of the gradients at
    once.
    :::
3.  [[3.]{.ltx_tag .ltx_tag_item}]{#A5.I2.i3}
    ::: {#A5.I2.i3.p1 .ltx_para}
    There are two places in the implementation where the transposition
    matters: (a) the choice of shard axis for the MLP matrices and
    (b) whether we compute the low-rank factorization for a gradient or
    its transpose. The former influences the bandwidth analysis, which
    we present in Section [[E.1]{.ltx_text
    .ltx_ref_tag}](#A5.SS1 "E.1 Bandwidth Analysis ‣ Appendix E Details for Distributed Optimization ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
    The latter influences the cost of the orthogonalization. Suppose
    that the gradient $G$ is $m \times n$ and its low-rank factors $P$
    and $Q^{t}$ are $m \times r$ and $r \times n$, respectively,
    with $r \ll {m,n}$. To make orthogonalization cheaper, we
    transpose $G$ appropriately so that $m \leqslant n$.
    :::

    ::: {#A5.I2.i3.p2 .ltx_para}
    At first glance, it may seem like a limitation that the NCCL
    all-gather and reduce-scatter primitives shard along axis 0 only. We
    may need to transpose some matrices before and after communication
    operations because of (a) and (b), which would require additional
    time and potentially special care to avoid out-of-memory errors. In
    fact, we never actually needed to do this. This is because we stored
    some of the parameters in their transposed formats and exploited the
    [transpose_a]{.ltx_text .ltx_font_typewriter} and
    [transpose_b]{.ltx_text .ltx_font_typewriter} parameters of the
    matrix multiplication kernels used in forward propagation,
    backpropagation, and steps (1)--(13) above. This allowed us to avoid
    explicit transposition while retaining the freedom to choose how to
    handle (a) and (b).
    :::
4.  [[4.]{.ltx_tag .ltx_tag_item}]{#A5.I2.i4}
    ::: {#A5.I2.i4.p1 .ltx_para}
    In step (12) above, we note that setting the error buffers to zero
    too often can cause performance regressions. We wanted to avoid
    doing this when resuming training from a checkpoint, which happens
    more frequently for larger jobs as it is likely that a machine will
    periodically fail. Naively, this would require uploading the error
    buffers from all of the machines along with the model checkpoints.
    Since we use a total of 128 machines for training, this would lead
    to 128 times greater storage usage, which is extremely wasteful.
    :::

    ::: {#A5.I2.i4.p2 .ltx_para}
    Fortunately, this is unnecessary, as error correction depends only
    on the sum of the error buffers. This property follows from
    linearity and the sequence of operations used by PowerSGD. Hence, it
    suffices to store the sums of the errors buffers taken across all
    GPUs with the same ordinal. When resuming from a checkpoint, we can
    divide the error buffers by the total number of machines and
    broadcast them.
    :::
:::
:::
:::

::: {#A6 .section .ltx_appendix}
## [Appendix F ]{.ltx_tag .ltx_tag_appendix}Details for Human Evaluation Experiments {#appendix-f-details-for-human-evaluation-experiments .ltx_title .ltx_title_appendix}

![[Figure 13: ]{.ltx_tag .ltx_tag_figure}Example task interface shown to
workers.](example_human_evals_task.png){#A6.F13.g1 .ltx_graphics
.ltx_centering .ltx_img_landscape width="598" height="334"}

::: {#A6.p1 .ltx_para}
We start with a list of $1000$ captions and generate one sample image
per model per caption. Captions and sample images are then used to
create $1000$ image comparison tasks per experiment, which we submitted
to Amazon's Mechanical Turk. Each task was answered by five distinct
workers. Workers were asked to compare two images and answer two
questions about them: (1) which image is most realistic, and (2) which
image best matches the shared caption. The experimental setup provided
to workers is shown in Figure [[13]{.ltx_text
.ltx_ref_tag}](#A6.F13 "Figure 13 ‣ Appendix F Details for Human Evaluation Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
One worker's answers were disqualified due to a high rate of
disagreement with other workers combined with a fast answer velocity
(with many submission times under 4 seconds); all other worker answers
were kept.
:::
:::

::: {#A7 .section .ltx_appendix}
## [Appendix G ]{.ltx_tag .ltx_tag_appendix}Zero-Shot Image-to-Image Translation {#appendix-g-zero-shot-image-to-image-translation .ltx_title .ltx_title_appendix}

![[(a) ]{.ltx_tag .ltx_tag_figure}"the exact same cat on the top as a
sketch on the bottom"](sketch_0.png){#A7.F14.sf1.g1 .ltx_graphics
.ltx_img_square width="70" height="70"}

::: {.ltx_flex_cell .ltx_flex_size_3}
![[(b) ]{.ltx_tag .ltx_tag_figure}"the exact same photo on the top
reflected upside-down on the bottom"](reflection_0.png){#A7.F14.sf2.g1
.ltx_graphics .ltx_img_square width="70" height="70"}
:::

::: {.ltx_flex_cell .ltx_flex_size_3}
![[(c) ]{.ltx_tag .ltx_tag_figure}"2 panel image of the exact same cat.
on the top, a photo of the cat. on the bottom, an extreme close-up view
of the cat in the photo."](close_up_0.png){#A7.F14.sf3.g1 .ltx_graphics
.ltx_img_square width="70" height="70"}
:::

::: {.ltx_flex_break}
:::

::: {.ltx_flex_cell .ltx_flex_size_3}
![[(d) ]{.ltx_tag .ltx_tag_figure}"the exact same cat on the top colored
red on the bottom"](red_0.png){#A7.F14.sf4.g1 .ltx_graphics
.ltx_img_square width="70" height="70"}
:::

::: {.ltx_flex_cell .ltx_flex_size_3}
![[(e) ]{.ltx_tag .ltx_tag_figure}"2 panel image of the exact same cat.
on the top, a photo of the cat. on the bottom, the cat with
sunglasses."](sunglasses_0.png){#A7.F14.sf5.g1 .ltx_graphics
.ltx_img_square width="70" height="70"}
:::

::: {.ltx_flex_cell .ltx_flex_size_3}
![[(f) ]{.ltx_tag .ltx_tag_figure}"the exact same cat on the top as a
postage stamp on the bottom"](postage_stamp_0.png){#A7.F14.sf6.g1
.ltx_graphics .ltx_img_square width="70" height="70"}
:::

[Figure 14: ]{.ltx_tag .ltx_tag_figure}Further examples of zero-shot
image-to-image translation.

::: {#A7.p1 .ltx_para}
Figure [[14]{.ltx_text
.ltx_ref_tag}](#A7.F14 "Figure 14 ‣ Appendix G Zero-Shot Image-to-Image Translation ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}
shows further examples of zero-shot image-to-image translation, which we
discussed in Section [[3.3]{.ltx_text
.ltx_ref_tag}](#S3.SS3 "3.3 Qualitative Findings ‣ 3 Experiments ‣ Zero-Shot Text-to-Image Generation"){.ltx_ref}.
We did not anticipate that this capability would emerge, and made no
modifications to the training procedure to encourage it.
:::
:::
:::

::: {.ltx_page_logo}
Generated on Fri Jul 4 15:05:41 2025 by [[L[a]{.ltx_font_smallcaps
style="position:relative; bottom:2.2pt;"}T[e]{.ltx_font_smallcaps
style="font-size:120%;position:relative; bottom:-0.2ex;"}]{style="letter-spacing:-0.2em; margin-right:0.1em;"}[XML]{style="font-size:90%; position:relative; bottom:-0.2ex;"}![Mascot
Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/){.ltx_LaTeXML_logo}
:::
:::
