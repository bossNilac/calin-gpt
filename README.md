# GPT From Scratch — A Guided Learning Journey

This repository is a structured attempt to understand how GPT-style language models work by **building the important pieces from scratch**.

It started as my own learning project, but the repository is organized so that **anyone who wants to follow the same path can use it as a curriculum**.

The goal is not to create the largest model possible, reproduce production-scale GPT systems, or hide complexity behind existing libraries.

The goal is to reach the point where the major components of a language model are no longer black boxes.

> If a component is used, the objective is to understand why it exists, how it works, and how to implement it.

---

## The Idea

Modern language models can look intimidating when encountered as one complete system.

But at a high level, the full pipeline is built from a sequence of understandable components:

```text
raw text
   ↓
tokenization
   ↓
token IDs
   ↓
training data
   ↓
embeddings
   ↓
transformer
   ↓
logits
   ↓
loss
   ↓
backpropagation
   ↓
trained model
   ↓
generation
```

This repository works through that pipeline incrementally.

Instead of beginning with a finished GPT implementation, each part is introduced separately, studied, implemented, tested, and eventually integrated into the full model.

The emphasis is on understanding the system from the bottom up.

---

## Who This Repository Is For

This repository is intended for people who:

- know how to program
- have some basic CS and mathematics background
- are interested in machine learning or language models
- want to understand Transformers beyond using a library
- prefer learning by implementing things themselves

You do not need to already be an ML expert.

The material is designed around gradually building the necessary understanding while progressing through the project.

If you want to follow the same path, begin with the first curriculum file and work through the repository in order.

---

## How to Use This Repository

Each stage of the project contains a curriculum and exercises.

For example:

```text
week1/
└── week1_curriculum_exercises.md
```

The curriculum explains:

- what concepts to study
- what components to implement
- exercises to complete
- questions you should be able to answer
- tests that can be used to verify understanding

The code in the repository represents my own implementation while going through that process.

You can use it in two ways.

### Follow the curriculum

The recommended approach is to read the curriculum and implement the exercises yourself before looking closely at my solution.

That preserves most of the learning value.

### Use the repository as a reference

If you are stuck, comparing your implementation against another implementation can be useful.

The important distinction is between:

```text
copying the solution
```

and:

```text
attempting the problem
→ identifying what you do not understand
→ comparing approaches
→ understanding the difference
```

The second approach is what this project is built around.

---

## Project Philosophy

### Understanding over abstraction

Libraries are useful, but they can hide exactly the mechanisms this project is trying to learn.

Where reasonable, important components are implemented manually rather than replaced with high-level equivalents.

PyTorch is still used for the infrastructure that is not the focus of the project, such as:

- tensors
- automatic differentiation
- GPU execution
- low-level numerical operations

The point is not to reinvent CUDA, BLAS, or an autograd engine.

The point is to understand the machine-learning system built on top of them.

---

### Understanding over model size

Parameter count is not the objective.

A small model that you can explain completely is more useful for this project than a large model assembled from components you do not understand.

Small models also make experimentation and debugging much easier.

---

### Implement first, compare later

When possible, the process is:

```text
study the concept
        ↓
derive the implementation
        ↓
write it
        ↓
test it
        ↓
debug it
        ↓
compare against a reference
```

Reference implementations are extremely useful, but they become more educational after you have already struggled with the problem yourself.

---

### Failure is part of the repository

Some implementations may be inefficient.

Some early approaches may later be replaced.

Some code may exist primarily because it helped expose a misunderstanding.

That is intentional.

This repository is not meant to pretend that everything was obvious from the beginning.

It documents the process of going from:

```text
"I know roughly what this does."
```

to:

```text
"I understand why this exists,
how it works,
how to implement it,
and how to debug it."
```

---

## What "From Scratch" Means Here

"From scratch" does **not** mean implementing an entire numerical computing stack.

It means implementing the important language-model components rather than importing a finished model.

The project may rely on PyTorch for tensor operations and autograd, while still manually implementing and understanding the systems built on top of those primitives.

The objective is to eventually understand the complete path from text to trained model and generated output.

---

## Learning Standard

A component is not considered understood simply because the code runs.

The standard throughout the project is closer to:

- Can I explain why this component exists?
- Can I describe the mathematics behind it?
- Can I predict its input and output shapes?
- Can I implement a basic version myself?
- Can I test whether it is correct?
- Can I identify what could go wrong?
- Can I explain how it interacts with the rest of the model?

That standard is more important than finishing quickly.

---

## Inspirations

The structure and philosophy of this project are not original inventions.

The learning path is heavily inspired by existing work from researchers, educators, and open-source projects that make neural networks and language models understandable from first principles.

### Andrej Karpathy — Neural Networks: Zero to Hero

A major inspiration for the teaching style of this repository.

The series demonstrates how increasingly sophisticated language models can be built by starting with simple systems and gradually introducing new concepts.

The emphasis on deriving components, examining tensor shapes, and understanding the mechanics rather than treating them as black boxes strongly influenced this project.

---

### Andrej Karpathy — nanoGPT

`nanoGPT` provides a compact and understandable implementation of GPT-style training.

It serves as an excellent reference for comparing design decisions after attempting to implement similar ideas independently.

Repository:

`karpathy/nanoGPT`

---

### Andrej Karpathy — build-nanogpt

`build-nanogpt` walks through the construction and training of a GPT-2-style model.

It is another important reference for understanding how the individual components eventually fit together into a complete training system.

Repository:

`karpathy/build-nanogpt`

---

### Attention Is All You Need

**Vaswani et al., 2017**

The original Transformer paper.

It introduced the attention-based architecture that modern GPT models build upon and remains one of the fundamental theoretical references for understanding Transformers.

---

### Language Models are Unsupervised Multitask Learners

**Radford et al., 2019**

The GPT-2 paper provides one of the primary architectural references for the kind of decoder-only language model this project ultimately works toward.

---

### PyTorch Documentation

PyTorch documentation is used extensively for understanding tensor operations, autograd behavior, optimization, and numerical details.

PyTorch implementations can also serve as useful correctness references for components implemented manually.

---

### Other Educational Material

Various university lectures, papers, technical articles, documentation, and open-source implementations naturally contribute to the learning process as well.

This repository should therefore be viewed as a **structured synthesis and hands-on implementation journey**, not as a novel architecture or original theory.

---

## Repository Structure

The repository is organized around progressive curriculum sections.

A simplified structure looks like:

```text
gpt-from-scratch/
│
├── README.md
│
├── week1/
│   ├── week1_curriculum_exercises.md
│   └── implementation...
│
├── week2/
│   ├── week2_curriculum_exercises.md
│   └── implementation...
│
├── week3/
│   └── ...
│
└── ...
```

Each `weekN_curriculum_exercises.md` file is intended to be usable independently by someone following the project.

The curriculum should ideally be read before looking at the corresponding implementation.

---

## If You Want to Follow Along

Start with:

```text
week1/week1_curriculum_exercises.md
```

Work through the concepts and exercises in order.

Try to write your own implementation before comparing it against the code in this repository.

Take notes.

Break things intentionally.

Inspect tensor shapes.

Test assumptions.

When something works, make sure you understand **why** it works.

If you reach the end with a smaller model but can explain the entire system confidently, that is a better outcome than reaching the end with a larger model you only partially understand.

---

## Final Goal

The final objective is not simply:

> "I built a GPT."

It is:

> "I understand how the major pieces of a GPT-style language model fit together because I studied, implemented, tested, and debugged them myself."

If this repository also helps someone else reach that point, then it has done more than document my own journey.
