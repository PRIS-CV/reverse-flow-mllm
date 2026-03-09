# Reversing the Flow: Generation-to-Understanding Synergy in Large Multimodal Models

> 
> Yujun Tong, Dongliang Chang, Zijin Yin,  Xintong Liu, Yuanchen Fang, Zhanyu Ma
> 
> Beijing University of Posts and Telecommunications


> The long-standing goal of multimodal AI is to build unified models in which visual understanding and visual generation mutually enhance one another. Despite recent works such as BAGEL, BLIP3o achieves remarkable progress; In practice, however, this unification remains one-directional: understanding routinely guides generation, yet how and why generation can support understanding is rarely investigated.We revisit this asymmetry and propose **Generation-to-Understanding (G→U) synergy**, where visual generation becomes an explicit intermediate reasoning step.  Our framework enables a model to perform controlled generative acts, such as detail enhancement, context expansion or structural visualisation, to produce self-generated \textbf{visual thoughts}, which are then fed back into the model to refine perception without retraining or external tools.Through a comprehensive evaluation on twelve benchmarks, this reversed information flow consistently improves multimodal understanding.  We show that generative fidelity bounds perceptual gain and that distinct families of edit prompts govern transfer efficiency.  We further analyse whether models can decide what to imagine. While they can produce plausible edits, these self-generated visual thoughts lack stable task alignment, revealing that current large multimodal models fall short of true self-reflection.This work exposes a missing mechanism in unified cognition and suggests that imagination is not the end of understanding but its beginning.
