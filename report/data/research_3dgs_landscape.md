# 3D Gaussian Splatting Technology Landscape for Cultural Heritage Preservation

Research compiled: 2026-03-31
Context: Academic report on applying 3DGS-based pipelines to cultural heritage digitisation and preservation.

---

## 1. 3D Gaussian Splatting (3DGS)

**Reference**: Kerbl, B., Kopanas, G., Leimkuehler, T., and Drettakis, G. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (SIGGRAPH 2023)*, 42(4), Article 139.

**GitHub**: https://github.com/graphdeco-inria/gaussian-splatting

**Key Contribution**: Introduced an explicit, point-based scene representation using anisotropic 3D Gaussians as rendering primitives. Each Gaussian is parameterised by position (mean), 3D covariance (factored as scale vector + rotation quaternion), opacity, and view-dependent colour encoded via spherical harmonics (SH). A differentiable tile-based rasteriser enables real-time rendering at >100 FPS while matching or exceeding NeRF quality (>30 dB PSNR on standard benchmarks). Training from Structure-from-Motion point clouds converges in minutes on a single GPU, compared to hours for neural radiance fields. Adaptive density control (clone, split, prune) during optimisation ensures the Gaussian population self-organises to represent scene geometry.

**Relevance to Cultural Heritage**: 3DGS offers a paradigm shift for heritage digitisation. Compared to photogrammetric meshes, 3DGS captures view-dependent appearance (specular stone, glazed ceramics, stained glass) with high fidelity. The compact, explicit representation (typically 1--5M Gaussians for a room-scale scene) enables interactive walkthroughs on consumer hardware. For heritage sites where re-visits are costly or impossible (conflict zones, deteriorating structures), 3DGS preserves visual appearance from casually captured video with minimal equipment. The standard PLY interchange format (62 float properties per Gaussian: position, SH coefficients, opacity, scale, rotation) ensures long-term archival without proprietary dependencies.

---

## 2. COLMAP Structure-from-Motion

**Reference**: Schoenberger, J.L. and Frahm, J.-M. (2016). "Structure-from-Motion Revisited." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016)*, pp. 4104--4113.

**GitHub**: https://github.com/colmap/colmap

**Key Contribution**: COLMAP is the de facto standard SfM and multi-view stereo (MVS) pipeline. It performs SIFT feature extraction (GPU-accelerated), exhaustive or sequential feature matching, incremental sparse reconstruction (bundle adjustment), and dense stereo fusion. Output: calibrated camera intrinsics and extrinsics, sparse 3D point cloud, and optionally dense depth maps. COLMAP 4.x adds CUDA-accelerated dense reconstruction and improved robustness for sequential video input.

**Limitations for Video Input**: COLMAP was designed for unordered photo collections. When applied to video, consecutive frames have high redundancy, which inflates matching time without improving reconstruction quality. Sequential matching (`--SiftMatching.max_num_neighbors`) mitigates this but misses loop-closure opportunities. Frame selection (every Nth frame, or quality-filtered) is essential. Rolling-shutter distortion from consumer cameras can degrade accuracy; pre-processing with `--ImageReader.camera_model OPENCV` or SIMPLE_RADIAL_FISHEYE helps. For very long video sequences (>1000 frames), hierarchical or vocabulary-tree matching is necessary to maintain tractable computation.

**Relevance to Cultural Heritage**: COLMAP is the standard first stage in heritage photogrammetry pipelines. Its robustness to varying lighting conditions and its ability to handle mixed camera types (DSLR + smartphone) make it suitable for field conditions. For walkthrough video of heritage interiors, sequential matching with loop closure at revisited locations provides reliable camera registration. The sparse point cloud serves as initialisation for 3DGS training.

---

## 3. LichtFeld Studio

**Reference**: LichtFeld Studio development team (MrNeRF). LichtFeld Studio: A Desktop Workstation for 3D Gaussian Splatting. Open-source, ongoing development (2024--present).

**GitHub**: https://github.com/MrNeRF/LichtFeld-Studio

**Key Contribution**: LichtFeld Studio is a C++/CUDA desktop application providing a complete GUI and programmatic interface for 3DGS workflows. Key features include:

- Integrated COLMAP SfM pipeline (headless, CUDA-accelerated)
- Multiple training strategies: MCMC, MRNF, IGS+ with pose optimisation
- 70+ MCP (Model Context Protocol) tools for programmatic control: training, camera manipulation, rendering, selection (rect/polygon/lasso/brush/click/description), export (PLY, SOG, SPZ, USD, HTML), scene graph operations, undo/redo, and GPU tensor access
- Scene graph with SPLAT, MESH, GROUP, and POINTCLOUD node types
- Import: PLY, SOG, SPZ, USD/USDA/USDC/USDZ, OBJ, FBX, glTF, GLB, STL, DAE
- Export: PLY, SOG, SPZ, USD, HTML viewer
- mesh2splat (EA, BSD-3 licensed): mesh-to-Gaussian conversion with PBR texture support
- PPISP correction for exposure, vignetting, colour response
- Python plugin system for extending functionality
- SplatReady video-to-COLMAP ingestion plugin

LichtFeld Studio exposes its full functionality via MCP, enabling agentic AI orchestration of the entire reconstruction pipeline.

**Relevance to Cultural Heritage**: LichtFeld Studio is the operational backbone for heritage 3DGS workflows. Its MCP automation interface enables repeatable, documented processing of heritage video captures. The scene graph allows hierarchical organisation of architectural elements (walls, columns, vaults, decorative features) as separate nodes. USD export provides interoperability with industry-standard DCC tools (Blender, Houdini, Omniverse). The Python plugin system enables domain-specific extensions such as condition assessment overlays or chronological layer visualisation.

---

## 4. gsplat

**Reference**: Ye, V., Turkulainen, M., the Nerfstudio Team (2024). "gsplat: An Open-Source Library for Gaussian Splatting." *arXiv preprint arXiv:2409.06765*.

**GitHub**: https://github.com/nerfstudio-project/gsplat

**Key Contribution**: gsplat is a standalone, production-grade CUDA rasterisation library for 3D Gaussian Splatting, developed by the Nerfstudio team. It provides a modular, well-documented Python/CUDA API decoupled from any specific training framework. Key technical contributions include:

- Efficient tile-based CUDA rasteriser with front-to-back alpha compositing
- Support for 2D and 3D Gaussian primitives
- Differentiable rendering with gradient computation for all Gaussian parameters
- Memory-efficient backward pass via per-tile sorting
- Integration with PyTorch autograd for seamless use in training loops
- Compression utilities for compact Gaussian storage

gsplat serves as the rendering backend for Nerfstudio's Splatfacto method and is adopted by numerous downstream research projects.

**Relevance to Cultural Heritage**: gsplat provides a stable, well-maintained rasterisation engine that can be embedded in custom heritage digitisation pipelines. Its PyTorch integration enables researchers to experiment with heritage-specific training objectives (e.g., multi-spectral SH coefficients, condition-aware opacity fields) without reimplementing the CUDA rasteriser. The library's permissive licensing (Apache-2.0) ensures it can be used in both academic and commercial heritage applications.

---

## 5. MILo (Mesh-In-the-Loop Gaussian Splatting)

**Reference**: Guedon, A., Gomez, D., Maruani, N., Gong, B., Drettakis, G., and Ovsjanikov, M. (2025). "MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction." *ACM Transactions on Graphics (SIGGRAPH Asia 2025)*, Journal Track.

**GitHub**: https://github.com/Anttwo/MILo

**Key Contribution**: MILo introduces a differentiable mesh-in-the-loop training scheme for 3DGS. During training, Gaussian pivot points are periodically Delaunay-triangulated to form a tetrahedral mesh. A learned signed distance function (SDF) is evaluated on this mesh, and the resulting surface is differentiably rendered alongside the Gaussians. Gradients from mesh rendering flow back to the Gaussian parameters, enforcing bidirectional consistency between the explicit Gaussian representation and the extracted mesh surface. Three mesh extraction methods are provided: learned SDF (highest quality), integrated opacity field, and regular TSDF. Output meshes are vertex-coloured PLY files at configurable resolution (250K to 14M Delaunay vertices).

MILo ships with three CUDA rasteriser backends (RaDe-GS, GOF, Mini-Splatting2) and requires CUDA 11.8, PyTorch 2.3.1, CGAL, and nvdiffrast. It accepts standard COLMAP datasets as input. Training runs for 18,000 iterations with mesh regularisation starting at iteration 8,001.

**License**: Gaussian-Splatting License (non-commercial, research and evaluation only).

**Relevance to Cultural Heritage**: MILo produces the highest-quality meshes from Gaussian representations, with significantly fewer vertices than competing methods at equivalent geometric fidelity. For heritage applications where polygonal mesh output is required (BIM integration, 3D printing of architectural elements, finite element analysis of structural components), MILo provides a direct path from video capture to accurate geometry. The mesh-in-the-loop training ensures that fine architectural details (mouldings, carvings, inscriptions) are preserved in the mesh rather than lost during post-hoc extraction. The non-commercial license restricts deployment but permits academic research and heritage documentation projects.

---

## 6. SuGaR (Surface-Aligned Gaussian Splatting)

**Reference**: Guedon, A. and Lepetit, V. (2024). "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)*.

**GitHub**: https://github.com/Anttwo/SuGaR (3,300+ stars)

**Key Contribution**: SuGaR is a post-hoc method that regularises trained 3D Gaussians to align with underlying surfaces, then extracts a watertight polygonal mesh via Poisson reconstruction. The pipeline:

1. Surface alignment: regularise Gaussians so their shortest axis aligns with the local surface normal
2. Poisson reconstruction: extract a watertight mesh from the aligned Gaussian centres and normals
3. Gaussian binding: re-bind Gaussians to mesh face centres
4. UV mapping: automatic UV parameterisation via Nvdiffrast
5. Texture baking: project SH colours onto the UV atlas to produce a diffuse texture map

SuGaR is the only published method that directly produces OBJ meshes with UV-mapped diffuse textures from 3D Gaussians. This makes its output directly compatible with standard rendering pipelines, USD materials (UsdPreviewSurface), and game engines.

**Relevance to Cultural Heritage**: SuGaR's textured mesh output is essential for heritage applications requiring standard polygon assets: museum exhibit interactives, VR walkthroughs with PBR materials, architectural plan extraction, and integration with heritage BIM systems. The UV-mapped textures preserve the photorealistic appearance captured by the Gaussians in a format compatible with all major DCC tools. For heritage objects requiring both geometric measurement and visual fidelity (e.g., painted architectural surfaces, decorated ceramics), SuGaR provides both in a single extraction pass. The trade-off is geometric over-smoothing from Poisson reconstruction, which can obscure very fine surface detail.

---

## 7. CoMe / SOF -- Confidence-Based and Sorted Opacity Field Mesh Extraction

**Reference (CoMe)**: Radl, T., Windisch, P., Kurz, D., Kohler, J., Steiner, M., and Steinberger, M. (2025). "CoMe: Confidence-Based Mesh Extraction from 3D Gaussians." *arXiv preprint arXiv:2603.24725*.

**Reference (SOF)**: Radl, T. et al. (2025). "SOF: Sorted Opacity Fields for Mesh Extraction from 3D Gaussians." *arXiv preprint arXiv:2506.19139*. SIGGRAPH Asia 2025.

**Reference (GOF)**: Yu, Z. et al. (2024). "Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes." *ACM SIGGRAPH Asia 2024*.

**GitHub (SOF)**: https://github.com/r4dl/SOF
**GitHub (CoMe)**: https://github.com/r4dl/CoMe (code forthcoming)
**GitHub (GOF)**: https://github.com/autonomousvision/gaussian-opacity-fields

**Key Contribution**: This family of methods (GOF -> SOF -> CoMe) represents the highest geometric accuracy for mesh extraction from Gaussians via Marching Tetrahedra on opacity fields.

- **GOF** (SIGGRAPH Asia 2024): Evaluates the Gaussian opacity field on an adaptive tetrahedral grid, identifies the 0.5-opacity level set, and extracts the isosurface via Marching Tetrahedra. Produces vertex-coloured meshes without UV maps.
- **SOF** (2025): Extends GOF with sorted opacity field evaluation (10x faster), better handling of unbounded scenes, and improved detail preservation.
- **CoMe** (2025): Adds per-primitive learnable confidence to the training process. Low-confidence Gaussians (reflections, sky, glass) are identified and suppressed during mesh extraction, reducing floater artifacts. Confidence-steered densification prevents over-population of ambiguous regions.

All three produce geometry-only PLY meshes with per-vertex colours. For textured output, a separate UV unwrapping (xatlas) and texture baking pass is required.

**Relevance to Cultural Heritage**: The GOF/SOF/CoMe family provides the most geometrically faithful mesh extraction available. For heritage contexts where measurement accuracy matters (archaeological site documentation, structural assessment, conservation planning), these methods preserve fine geometric detail that Poisson-based approaches (SuGaR) smooth away. CoMe's confidence mechanism is particularly valuable for heritage interiors with reflective surfaces (polished marble, gilt surfaces, stained glass windows), where standard methods produce mesh artifacts. The lack of UV textures is a minor limitation since texture baking can be performed as a post-process using the training images.

---

## 8. SAM3 / SAM2 -- Segment Anything Models

**Reference (SAM)**: Kirillov, A. et al. (2023). "Segment Anything." *IEEE/CVF International Conference on Computer Vision (ICCV 2023)*.

**Reference (SAM 2)**: Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos." *arXiv preprint arXiv:2408.00714*. Meta AI.

**Reference (SAM 3)**: Meta AI (2025). "Segment Anything Model 3." Apache-2.0 license.

**GitHub (SAM)**: https://github.com/facebookresearch/segment-anything
**GitHub (SAM 2)**: https://github.com/facebookresearch/sam2

**Key Contribution**:

- **SAM** (ICCV 2023): Foundation model for promptable image segmentation. Trained on 11M images with 1.1B masks. Accepts point, box, or mask prompts. Produces class-agnostic instance masks.
- **SAM 2** (2024): Extends SAM to video with a memory-based architecture for temporally consistent mask tracking across frames. Streaming inference enables real-time video segmentation. The memory mechanism propagates object identity across occlusions and viewpoint changes.
- **SAM 3** (2025): Unified model for detection, segmentation, and tracking using both text and visual prompts across 4M concepts. No prompt engineering required -- objects are described in natural language (e.g., "Corinthian column capital", "mosaic tessera", "carved stone lintel"). SAM 3.1 adds Object Multiplex for multi-object tracking across video frames.

The SAM family underpins most 3DGS segmentation methods. Gaussian Grouping, SAGA, and OpenGaussian all lift SAM 2D masks to 3D Gaussians via multi-view consistency.

**Relevance to Cultural Heritage**: SAM3's text-prompted segmentation with 4M concepts is transformative for heritage documentation. Operators can segment architectural elements by name ("flying buttress", "rose window", "pilaster") without manual mask annotation. Multi-view consistency (via SAM2's memory mechanism or SAM3's Object Multiplex) ensures that an element identified in one video frame maintains its identity across the entire walkthrough. This enables automatic decomposition of a heritage scene into architecturally meaningful components -- a prerequisite for structured USD scene graphs and element-level condition assessment. The Apache-2.0 license permits use in heritage preservation projects without restriction.

---

## 9. Hunyuan3D

**Reference**: Tencent (2025). "Hunyuan3D 2.0: Scaling Diffusion Models for High-Resolution Textured 3D Asset Generation." Tencent Hunyuan Community License.

**GitHub**: https://github.com/Tencent/Hunyuan3D-2
**HuggingFace**: https://huggingface.co/tencent/Hunyuan3D-2

**Key Contribution**: Hunyuan3D 2.0 is a two-stage 3D asset generation system:

1. **Shape generator (DiT, 1.1B parameters)**: Generates 3D geometry from single or multi-view images via a diffusion transformer operating in a 3D VAE latent space with octree voxelisation. Multi-view conditioning (front/left/back/right) provides superior geometric accuracy.
2. **Texture synthesizer (Paint, 1.3B parameters)**: Generates UV-mapped diffuse textures for the extracted mesh. A delight model separates lighting from material properties for PBR output.

The multi-view variant (Hunyuan3D-2mv) accepts 4 canonical views and produces significantly more accurate geometry than single-image approaches. Output: textured GLB/OBJ meshes.

VRAM requirements: 6 GB (shape only), 16 GB (shape + texture). Models total approximately 75 GB on HuggingFace.

**Relevance to Cultural Heritage**: Hunyuan3D provides an alternative mesh reconstruction pathway for heritage objects extracted from Gaussian scenes. By rendering 4 canonical views from per-object Gaussians and feeding them to Hunyuan3D, heritage objects receive high-quality textured meshes even when TSDF or Poisson methods struggle (e.g., thin structures, hollow objects, complex topology). For isolated heritage artefacts (sculptures, pottery, architectural fragments), Hunyuan3D's learned geometric priors can complete partially observed geometry -- valuable when physical access constraints limit viewpoint coverage. The community license permits non-commercial heritage use.

---

## 10. OpenUSD (Universal Scene Description)

**Reference**: Pixar Animation Studios (2016--present). "OpenUSD: Universal Scene Description." Open-source (Modified Apache-2.0 License).

**Documentation**: https://openusd.org/
**GitHub**: https://github.com/PixarAnimationStudios/OpenUSD

**Key Contribution**: OpenUSD is a scene description framework for authoring, composing, and reading hierarchical 3D scene data. Key concepts:

- **Prims**: Scene graph nodes (Xform, Mesh, Camera, Light, Material, custom types)
- **Composition arcs**: References, payloads, variant sets, inherits, specialises -- enabling non-destructive layering of scene data from multiple sources
- **Variant sets**: Multiple representations of the same prim (e.g., Gaussian splat view vs. polygonal mesh view of the same heritage object)
- **Materials**: UsdPreviewSurface for PBR materials with diffuse, metallic, roughness, normal maps
- **Python API** (`pxr` package): Programmatic scene construction via `Usd`, `UsdGeom`, `UsdShade`, `Sdf` modules
- **File formats**: `.usda` (ASCII, human-readable), `.usdc` (binary, compact), `.usdz` (zipped archive for distribution)

The August 2025 Khronos `KHR_gaussian_splatting` glTF extension establishes a complementary standard for Gaussian splat interchange, but OpenUSD remains the scene-level composition format.

**Relevance to Cultural Heritage**: OpenUSD provides the ideal scene description format for heritage digital twins. Its hierarchical scene graph maps naturally to architectural decomposition (site > building > room > wall > decorative element). Variant sets enable dual representation: a photorealistic Gaussian splat variant for visual inspection and a polygonal mesh variant for measurement and analysis, selectable without duplicating data. Composition arcs allow incremental documentation -- a baseline scan can be extended with condition assessments, conservation records, or new captures as non-destructive layers. The Python API enables automated scene assembly from pipeline outputs. USD's adoption by NVIDIA Omniverse, Apple visionOS, and industry DCC tools (Blender, Houdini, Maya) ensures long-term interoperability for heritage archives.

---

## 11. WebGL/WebGPU Gaussian Splat Viewers

**Reference (antimatter15)**: Kevin Kwok (2023). "WebGL 3D Gaussian Splat Viewer." https://github.com/antimatter15/splat

**Reference (gsplat.js)**: Dylan Ebert et al. (2024). "gsplat.js: JavaScript Gaussian Splatting Library." https://github.com/huggingface/gsplat.js

**Reference (model-viewer)**: Google (2019--present). "`<model-viewer>` Web Component for 3D/AR." https://github.com/google/model-viewer

**Key Contribution**: Browser-based 3DGS rendering has matured rapidly:

- **antimatter15/splat**: One of the first WebGL Gaussian splat viewers. Demonstrates real-time rendering of PLY Gaussian scenes in the browser via WebGL 2.0 compute shaders emulated through fragment shaders.
- **gsplat.js** (HuggingFace): Production-grade JavaScript library for Gaussian splatting in the browser. Supports PLY and SPZ formats, progressive loading, and camera controls. Used by HuggingFace Spaces for 3DGS model hosting.
- **PlayCanvas SuperSplat**: WebGPU-accelerated viewer and editor for Gaussian splats. Supports the compressed SPZ format for efficient streaming. https://github.com/playcanvas/supersplat
- **Luma AI UnboundedGS**: Commercial WebGL viewer supporting compressed Gaussian scenes with level-of-detail streaming.
- **`<model-viewer>`** (Google): Web component supporting glTF/GLB mesh viewing with AR placement. Used in the Gaussian Toolkit web interface for mesh preview.

The Khronos `KHR_gaussian_splatting` glTF extension (August 2025) standardises Gaussian splat encoding within glTF files, enabling any glTF-compatible viewer to display Gaussian scenes.

**Relevance to Cultural Heritage**: Browser-based viewers eliminate the barrier to accessing heritage 3D documentation. Researchers, conservators, and the public can explore digitised heritage sites without installing specialised software. For museums, WebGPU viewers embedded in collection pages provide interactive 3D object inspection. The SPZ compressed format and progressive loading enable streaming of large heritage scenes over bandwidth-constrained connections. IIIF (International Image Interoperability Framework) integration with 3DGS viewers is an emerging area for heritage institutions.

---

## 12. Unreal Engine 3DGS Plugins

**Reference (UE 5.4+ Gaussian Splatting)**: Various community implementations and Epic Games experimental features.

**Key Implementations**:

- **Luma AI Unreal Plugin**: Commercial plugin for importing and rendering Gaussian splat scenes in UE5. Supports Luma's compressed format. Real-time rendering via custom material shaders.
- **3D Gaussian Splatting for Unreal Engine (unrealgs)**: Open-source plugin by Amir Semmo et al. Imports PLY Gaussian scenes as Niagara particle systems. https://github.com/xverse-engine/XV3DGS-UEPlugin
- **Cesium + 3DGS**: Cesium for Unreal supports streaming 3D Tiles with Gaussian splat payloads for large-scale outdoor heritage scenes.
- **Gaussian Splat UE Plugin (Luma)**: https://lumalabs.ai/unreal-engine

As of early 2026, Unreal Engine does not have native first-party 3DGS support, but the community plugin ecosystem is maturing. The typical approach encodes Gaussian attributes as particle system parameters and renders via instanced billboards or custom compute shaders.

**Relevance to Cultural Heritage**: Unreal Engine is increasingly used for heritage visualisation, virtual museum exhibits, and architectural walkthroughs. 3DGS plugins enable heritage scenes reconstructed from video to be directly imported into UE5 for interactive real-time experiences. Combined with UE5's Lumen global illumination and Nanite geometry system for mesh components, hybrid Gaussian + mesh scenes can achieve photorealistic heritage walkthroughs. For accessibility applications (virtual heritage visits for mobility-impaired users, remote education), UE5's cross-platform deployment (PC, console, VR headsets, mobile via streaming) maximises reach.

---

## 13. TSDF Volumetric Fusion

**Reference**: Curless, B. and Levoy, M. (1996). "A Volumetric Method for Building Complex Models from Range Images." *Proceedings of SIGGRAPH 1996*, pp. 303--312.

**Reference (KinectFusion)**: Newcombe, R.A. et al. (2011). "KinectFusion: Real-Time Dense Surface Mapping and Tracking." *IEEE ISMAR 2011*.

**Key Implementation**: Open3D `ScalableTSDFVolume` -- https://github.com/isl-org/Open3D

**Key Contribution**: Truncated Signed Distance Function (TSDF) volumetric fusion is the classical approach to dense surface reconstruction from depth maps. The algorithm:

1. Discretise 3D space into a voxel grid
2. For each depth map, compute the signed distance from each voxel to the observed surface
3. Truncate the signed distance to a narrow band around the surface
4. Average (fuse) TSDF values across all depth maps via running weighted average
5. Extract the zero-crossing isosurface via Marching Cubes

**Limitations**:
- Resolution is fixed by voxel size; fine details require very high-resolution grids (memory-intensive)
- Bounded volume: the grid must encompass the entire scene a priori
- No view-dependent appearance: produces geometry-only meshes with vertex colours from averaged image projections
- Depth map noise propagates directly to surface artifacts
- For 3DGS workflows, depth maps must be rendered from trained Gaussians (not directly observed), introducing an additional source of error

**Relevance to Cultural Heritage**: TSDF fusion remains a practical baseline for heritage mesh extraction. Its implementation simplicity (available in Open3D, PCL, and custom CUDA kernels) and predictable behaviour make it suitable for production pipelines where robustness is prioritised over cutting-edge quality. For large-scale heritage scenes (entire rooms, building exteriors), TSDF fusion with Open3D's scalable volume can process scenes that exhaust the memory of Poisson or Marching Tetrahedra methods. Typical results: 20K--200K vertices depending on voxel resolution, extraction time 2--15 minutes. The fixed grid resolution is a poor fit for heritage scenes with mixed scales (large flat walls + fine carved details); adaptive methods (SOF, MILo) address this limitation.

---

## 14. Poisson Surface Reconstruction vs Marching Cubes

**Reference (Poisson)**: Kazhdan, M., Bolitho, M., and Hoppe, H. (2006). "Poisson Surface Reconstruction." *Eurographics Symposium on Geometry Processing (SGP 2006)*, pp. 61--70.

**Reference (Screened Poisson)**: Kazhdan, M. and Hoppe, H. (2013). "Screened Poisson Surface Reconstruction." *ACM Transactions on Graphics*, 32(3), Article 29.

**Reference (Marching Cubes)**: Lorensen, W.E. and Cline, H.E. (1987). "Marching Cubes: A High Resolution 3D Surface Construction Algorithm." *ACM SIGGRAPH Computer Graphics*, 21(4), pp. 163--169.

**Key Implementation (Poisson)**: Open3D `create_from_point_cloud_poisson` -- https://github.com/isl-org/Open3D
**Key Implementation (Marching Cubes)**: scikit-image `marching_cubes`, Open3D TSDF extraction

**Comparison**:

| Property | Poisson Reconstruction | Marching Cubes (on TSDF/SDF) |
|----------|----------------------|------------------------------|
| Input | Oriented point cloud (positions + normals) | Volumetric scalar field (TSDF or opacity) |
| Output | Watertight manifold mesh | Isosurface mesh (may have boundaries) |
| Topology | Always closed (watertight) | Follows field topology (may be open) |
| Smoothness | Inherently smooth (global variational) | Depends on field resolution |
| Fine detail | Tends to over-smooth | Preserves detail at grid resolution |
| Memory | O(N) where N = point count | O(R^3) where R = grid resolution |
| Heritage use | Good for isolated objects (sculptures, pottery) | Better for architectural interiors |

**Marching Tetrahedra** (used by SOF, CoMe, MILo) is an alternative to Marching Cubes that avoids topological ambiguity cases. It operates on a tetrahedral mesh rather than a regular grid, enabling adaptive resolution that concentrates detail where the scene requires it.

**Relevance to Cultural Heritage**: Both methods serve different heritage digitisation needs. Poisson reconstruction (used by SuGaR, Nerfstudio's `ns-export poisson`) produces clean, watertight meshes ideal for 3D printing heritage replicas and museum display copies. Its over-smoothing tendency can obscure tool marks, weathering patterns, and fine inscriptions that are archaeologically significant. Marching Cubes on TSDF (used by Nerfstudio's `ns-export tsdf`, Open3D fusion) preserves more surface detail but may produce meshes with holes at unobserved regions. For comprehensive heritage documentation, both methods should be applied: Poisson for presentation-quality models and Marching Cubes/Tetrahedra for measurement-grade geometry.

---

## 15. Blender Integration with 3DGS Workflows

**Reference**: Blender Foundation (2024). Blender 4.x / 5.x. Open-source 3D creation suite. https://www.blender.org/

**Key Integration Points**:

- **USD Import/Export**: Blender 3.5+ includes native USD I/O. Import USD scenes assembled from pipeline outputs; export refined scenes for archival.
- **Mesh Import**: Assimp-based import of OBJ, PLY, FBX, glTF/GLB meshes produced by SuGaR, MILo, TSDF extraction.
- **Texture Baking**: Cycles renderer can bake diffuse, normal, ambient occlusion, and emission maps from high-poly meshes onto UV-mapped low-poly meshes. Essential for vertex-coloured meshes (GOF/SOF output) that need UV textures for USD materials.
- **Material Assignment**: Shader Editor enables PBR material creation (UsdPreviewSurface-compatible) for heritage assets.
- **Scene Assembly**: Python scripting (`bpy`) enables automated scene graph construction, camera placement from COLMAP extrinsics, and batch rendering.
- **Rendering**: Cycles (physically-based path tracer, GPU-accelerated) and EEVEE (real-time) for heritage visualisation and publication renders.
- **MCP Integration**: Blender can be controlled via MCP socket server for agentic workflow orchestration.
- **Gaussian Splatting Addons**: Community addons (e.g., blender-gaussian-splatting by maximeheckel) enable direct import and viewing of Gaussian PLY files within Blender's viewport.

**Relevance to Cultural Heritage**: Blender serves as the scene assembly and texture baking hub in heritage 3DGS pipelines. Its open-source nature and zero licensing cost make it accessible to heritage organisations with limited budgets. The Python scripting API enables automated processing of dozens of heritage objects through standardised material assignment and UV baking workflows. Blender's USD export produces archive-quality scene files compatible with NVIDIA Omniverse and Apple visionOS for institutional distribution. For heritage publication, Cycles rendering produces photorealistic images for reports, catalogues, and public engagement materials.

---

## Summary: Technology Readiness for Cultural Heritage Preservation

| Technology | TRL | Heritage Readiness | Key Gap |
|-----------|-----|-------------------|---------|
| 3DGS (Kerbl et al.) | 8 | High | No inherent measurement scale |
| COLMAP SfM | 9 | Very High | Video input requires frame selection |
| LichtFeld Studio | 6 | Medium | USD mesh export still developing |
| gsplat | 7 | High | Rendering only, no scene management |
| MILo | 5 | Medium | Non-commercial license |
| SuGaR | 7 | High | Over-smooths fine heritage detail |
| CoMe/SOF | 5 | Medium | No UV textures; CoMe code unreleased |
| SAM3/SAM2 | 7 | High | Heritage-specific concept vocabulary needed |
| Hunyuan3D | 6 | Medium | Geometry completion may hallucinate detail |
| OpenUSD | 9 | Very High | No native Gaussian prim type (community extension) |
| WebGL/WebGPU viewers | 7 | High | Large scene streaming needs optimisation |
| UE5 3DGS plugins | 4 | Low | No native support; plugin fragmentation |
| TSDF fusion | 9 | Very High | Fixed resolution, poor for mixed-scale scenes |
| Poisson/Marching Cubes | 9 | Very High | Poisson over-smooths; MC requires good TSDF |
| Blender integration | 8 | High | Manual steps for 3DGS-specific workflows |

**Technology Readiness Level (TRL)**: 1 = basic principles, 5 = technology validated, 7 = prototype demonstrated, 9 = proven in operational environment.

---

## Consolidated Reference List

1. Curless, B. and Levoy, M. (1996). "A Volumetric Method for Building Complex Models from Range Images." *SIGGRAPH 1996*.
2. Kazhdan, M., Bolitho, M., and Hoppe, H. (2006). "Poisson Surface Reconstruction." *SGP 2006*.
3. Lorensen, W.E. and Cline, H.E. (1987). "Marching Cubes: A High Resolution 3D Surface Construction Algorithm." *ACM SIGGRAPH 1987*.
4. Newcombe, R.A. et al. (2011). "KinectFusion: Real-Time Dense Surface Mapping and Tracking." *IEEE ISMAR 2011*.
5. Kazhdan, M. and Hoppe, H. (2013). "Screened Poisson Surface Reconstruction." *ACM TOG*, 32(3).
6. Schoenberger, J.L. and Frahm, J.-M. (2016). "Structure-from-Motion Revisited." *CVPR 2016*.
7. Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG (SIGGRAPH 2023)*, 42(4).
8. Kirillov, A. et al. (2023). "Segment Anything." *ICCV 2023*.
9. Guedon, A. and Lepetit, V. (2024). "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction." *CVPR 2024*.
10. Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos." *arXiv:2408.00714*.
11. Ye, V. et al. (2024). "gsplat: An Open-Source Library for Gaussian Splatting." *arXiv:2409.06765*.
12. Yu, Z. et al. (2024). "Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes." *ACM SIGGRAPH Asia 2024*.
13. Guedon, A. et al. (2025). "MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction." *ACM TOG (SIGGRAPH Asia 2025)*.
14. Radl, T. et al. (2025). "SOF: Sorted Opacity Fields for Mesh Extraction from 3D Gaussians." *arXiv:2506.19139*.
15. Radl, T. et al. (2025). "CoMe: Confidence-Based Mesh Extraction from 3D Gaussians." *arXiv:2603.24725*.
16. Meta AI (2025). "Segment Anything Model 3." Apache-2.0.
17. Tencent (2025). "Hunyuan3D 2.0: Scaling Diffusion Models for High-Resolution Textured 3D Asset Generation." Tencent Hunyuan Community License.
18. Pixar Animation Studios (2016--present). "OpenUSD: Universal Scene Description." Modified Apache-2.0.
