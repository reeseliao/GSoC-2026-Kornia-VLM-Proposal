# GSoC 2026 Proposal: High-Performance Local VLM Inference for Bubbaloop & Kornia-RS

**Organization:** Kornia

**Applicant:** Yudan (Reese) Liao 

**Target Project:** 2. Local vision-language model application on Bubbaloop

**Mentors:** Edgar Riba, Miquel Farré

---
## About Me

I am Yudan (Reese) Liao, a second-year BSc Applied Computing student at HSUHK (top 10% of cohort), specializing in AI Infrastructure and Systems Engineering. 

My engineering trajectory is rooted in computer vision and systems-level AI. I previously served as a Research Assistant optimizing PointNet++ for 3D point cloud transmission in autonomous driving, and placed in the global top 8% in a Kaggle LLM fine-tuning competition. 

**The Shift to C++/Rust & Edge AI:**
Most recently, while architecting an offline multimodal RAG engine (LLaVA + Gemma 2 + FAISS) for local video search, I directly confronted the memory bloat and GIL bottlenecks that make pure Python an architectural dead end for production-grade edge AI. This painful profiling experience drove my transition into building high-performance, native pipelines. 

Rust Proficiency:
I have working proficiency in Rust, with hands-on experience navigating the ort ONNX Runtime bindings through PR #816 to kornia-rs. I am comfortable with Rust's ownership model and async patterns, and I am actively deepening my systems-level Rust through this project.

My goal is to build the underlying native pipelines that make AI actually deployable on constrained hardware — not just application wrappers.
## 1. Project Abstract & Motivation
The `kornia-vlm` and Bubbaloop ecosystems are pioneering the deployment of local Vision-Language Models (VLMs) for robotics and spatial AI. While the current infrastructure leverages the `candle` framework for pure-Rust inference, deploying heavy models like `Qwen2-VL` to embedded edge hardware (e.g., NVIDIA Jetson Orin) demands extreme, bare-metal hardware acceleration. 

This project aims to bridge this gap by architecting a high-performance **ONNX Runtime (ORT) backend powered by TensorRT Execution Providers**. By expanding `kornia-rs` into this currently unexplored territory, this project will enable hardware-accelerated, quantized (INT4/INT8) VLM inference. The ultimate deliverable is a seamless C++/Rust pipeline directly powering a real-time visual Q&A and scene understanding application within the Bubbaloop environment.

**Motivation: Why This Project**
My passion for this project stems directly from my recent engineering obsession: squeezing maximum inference speed out of constrained hardware (AI-PCs and Edge Devices). I noticed that while Kornia provides excellent vision primitives, the ONNX/TensorRT VLM pipeline is currently a greenfield project waiting to be optimized.

When architecting a local multimodal semantic video search engine recently, I learned the hard way that CPU-bound inference or unoptimized backends introduce severe latency (e.g., ~5.1s/frame for VLMs). To solve this within the Kornia ecosystem, I recently dove into the `kornia-rs` codebase and developed the dynamic device routing logic for ONNX Runtime (handling CUDA/CPU fallbacks), familiarizing me with ort's execution provider API and Kornia's image tensor scaling ops (`kornia::image::ops::cast_and_scale`).

This hands-on experience proved to me that the true potential of Bubbaloop on edge devices lies in bypassing software overhead and directly hooking into NVIDIA's TensorRT engines. I am eager to combine my C++/Rust proficiency and my experience with compiled model caching to build the absolute fastest local VLM experience for the Kornia community.
## 2. Technical Architecture & Engineering Strategy

To achieve real-time VLM inference on edge devices like the Jetson Orin, the architecture must strictly avoid CPU bottlenecks and redundant memory allocations. The proposed system is decoupled into three high-performance layers:

### 2.1. The Execution Layer: TensorRT via `ort` Bindings
Building directly upon my recent work in `kornia-vlm` (PR #816, which implemented dynamic device routing for CUDA/CPU), I will extend the `kornia-rs` engine to fully support the **TensorRT Execution Provider (TRT EP)**.
* **Engine Caching:** TensorRT model compilation (building the `.engine` plan) is notoriously slow and can cause a massive cold-start penalty on edge devices. I will implement a persistent `.engine` cache loading mechanism within the Rust backend. Upon the first initialization of `Qwen2-VL`, the serialized TensorRT engine will be saved to disk, reducing subsequent startup times from minutes to mere seconds.
* **Quantization Strategy:** To fit the 16GB/8GB unified memory of typical Jetson devices, the ONNX models will be strictly quantized to INT4/INT8 (using tools like vLLM or ONNX Runtime quantization) before being fed into the TensorRT engine.

### 2.2. Memory Orchestration: The Zero-Copy Pipeline
The most common point of failure in edge AI is the "Data Starvation" of the GPU caused by slow CPU pre-processing. 
* **Optimized Pre-processing:** I will heavily leverage Kornia's native tensor operations. Instead of copying image arrays multiple times across the Rust heap, operations like `kornia::image::ops::cast_and_scale` and HWC-to-CHW permutations will be optimized to write directly into pre-allocated, continuous memory buffers that the ONNX Runtime `Value::from_array` can ingest with zero or minimal copying.
* **Continuous Streaming:** For video inputs from Bubbaloop, the extraction and inference pipelines will be decoupled using a lock-free asynchronous channel (e.g., `tokio::sync::mpsc` or a custom ring buffer). This ensures the camera ingestion loop is never blocked by the VLM inference loop.

### 2.3. The Application Layer: Bubbaloop FFI Bridge
`Bubbaloop` requires a responsive UI. The heavy VLM inference must not freeze the application.
* **Async FFI Interface:** I will expose a non-blocking asynchronous Rust API from `kornia-vlm` to the Bubbaloop application.
* **Streamed Generation:** Instead of waiting for the entire VLM response to generate, I will implement a token-streaming mechanism using `ort`'s generator APIs, passing text chunks back to the Bubbaloop UI in real-time, drastically reducing the perceived Time-To-First-Token (TTFT) for the user.

## 3. Implementation Plan & 12-Week Timeline (350 Hours)

My development philosophy strictly follows agile principles: **Make it work, make it fast, make it beautiful.** The 12-week timeline is structured to deliver a minimum viable pipeline early, leaving ample time for extreme hardware optimizations.

### Phase 1: Foundation & Backend Fortification (Weeks 1 - 4)
**Goal:** Solidify the `kornia-rs` execution engine to robustly support TensorRT and zero-copy memory operations.
* **Week 1:** Expand upon the base routing logic established in PR #816. Implement comprehensive lifecycle management for the TensorRT Execution Provider within the ort Rust bindings, ensuring safe fallback mechanisms and robust memory allocation.
* **Week 2:** Develop the Zero-copy memory allocator in Rust. Map Kornia's image tensors directly to ONNX Runtime's expected memory layout without heap duplication.
* **Week 3:** Introduce the `.engine` compilation and caching mechanism. Write the serialization logic to save compiled TensorRT plans to the local disk.
* **Week 4 (Milestone 1):** Deliver a standalone, headless CLI tool in `kornia-rs` capable of loading a dummy ONNX model via TensorRT with verified cache hits.

### Phase 2: Qwen2-VL Integration & Optimization (Weeks 5 - 8)
**Goal:** Successfully run Qwen2-VL-2B-Instruct through the new backend and crush the latency bottlenecks.
* **Week 5:** Integrate the INT4/INT8 quantized Qwen2-VL model. Handle the specific tokenization and multi-modal embedding requirements within the Rust pipeline.
* **Week 6:** Implement the Token-Streaming generator API. Ensure text chunks can be yielded asynchronously rather than waiting for full sequence generation.
* **Week 7:** Rigorous Hardware Profiling & Cross-Compilation. Since edge devices have strict memory limits, I will simulate edge constraints on local AI-PC hardware (e.g., artificially capping RAM/VRAM availability) to identify CPU pre-processing bottlenecks (like `cast_and_scale`). Simultaneously, I will configure the cross-compilation toolchains (`aarch64-unknown-linux-gnu`) to ensure the Rust backend compiles flawlessly for ARM64/Jetson targets.
* **Week 8 (Milestone 2):** Achieve End-to-End VLM inference in the headless CLI. Generate a preliminary Benchmark Report showing Latency, TTFT (Time-To-First-Token), and Memory Footprint.

### Phase 3: Bubbaloop UI Integration & Delivery (Weeks 9 - 12)
**Goal:** Expose the blistering-fast backend to the user through a seamless desktop/edge application.
* **Week 9:** Build the FFI (Foreign Function Interface) or Async IPC bridge between the `kornia-vlm` Rust core and the Bubbaloop application layer.
* **Week 10:** Develop the interactive UI components in Bubbaloop for Video Q&A and real-time Scene Captioning, connecting them to the streaming token generator.
* **Week 11:** Cross-platform verification and UI bug squashing. I will verify the entire `kornia-vlm` to Bubbaloop pipeline under constrained memory profiles locally. I will also coordinate with the mentors to test the pre-compiled ARM binaries on actual target edge hardware (e.g., Jetson Orin) to ensure the `.engine` cache mechanism works as expected on NVIDIA embedded GPUs.
* **Week 12 (Milestone 3 & Final):** Finalize all documentation, polish the codebase, and record the high-quality video demo.

---

## 4. Expected Deliverables

1. **`kornia-rs` Core PRs:** A production-ready TensorRT backend with engine caching and zero-copy data ingestion.
2. **Bubbaloop Integration:** A working, interactive visual Q&A application running local VLMs.
3. **Comprehensive Benchmarks:** Detailed metrics comparing pure CPU (`candle`) vs. CUDA vs. TensorRT performance for VLMs.
4. **Documentation & Demo:** Developer guides for the new backend and a final video demonstration.

---


## 5. Communication & Availability Plan

Open source is inherently collaborative, and I believe proactive communication is just as critical as code quality.

* **Timezone & Availability:** I am based in Shenzhen / Hong Kong (UTC+8). I can comfortably commit to **30-35 hours per week** for this 350-hour large project during the GSoC coding period.
* **Daily/Weekly Sync:** I plan to push code frequently (at least every 2-3 days) to allow for iterative, bite-sized code reviews rather than massive, unreviewable PRs. I am highly responsive on Discord (`#kornia-rs` or DMs) and am available for a weekly video sync with my mentors (Edgar / Miquel) at a time that suits European/US timezones.
* **Documentation:** I will maintain a public weekly dev-log on GitHub Discussions or a personal blog, documenting technical hurdles (e.g., specific `ort` binding issues or FFI memory leaks) and their solutions, ensuring the community benefits from the development process.

---

