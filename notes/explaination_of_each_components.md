## Why do we need Feature Extractor ?

The Multi-Scale Feature Encoder (CNN backbone with C3, C4, C5 outputs at different resolutions) is essential because objects in traffic scenes vary drastically in size—a nearby car might occupy 500×300 pixels while a distant license plate is only 60×20 pixels—and different tasks require different spatial granularities: coarse features (C5, 1/32 resolution) capture global context for vehicle detection and scene understanding, while fine features (C3, 1/8 resolution) preserve detailed edges and textures critical for segmenting driveway boundaries and recognizing small license plate characters. 


## Why do we need Transformer Encoder ?
The Transformer Encoder is then necessary because CNNs only capture local patterns within their receptive fields, but our tasks require global reasoning: detecting that a stopped vehicle (spatial reasoning across the entire frame) overlaps with a driveway (spatial relationship between distant image regions), understanding that a license plate belongs to a specific vehicle 200 pixels away (long-range association), and tracking vehicles across occlusions where appearance features from one part of the image must be matched with motion patterns elsewhere. The Transformer's self-attention mechanism allows every spatial location to attend to every other location, enabling queries like "where are ALL vehicles in this scene?" and "which plate belongs to which vehicle?", creating a unified global representation that all five decoders can then interpret through their task-specific lenses—something purely convolutional or purely local processing cannot achieve.

## Why do we need Learnable Queries for each decoder ?
**Learnable queries are the fundamental "questions" each decoder asks about the image, and they're necessary because decoders need task-specific starting points to extract relevant information from the generic encoder features.** Without queries, the decoder would have no way to specify *what* it's looking for—the encoder provides a rich spatial feature map of the entire scene, but the decoder needs to probe it with specific intentions: "where are the objects?", "what are the segmentation masks?", "where are the plates?". Queries are **randomly initialized embeddings** (vectors of dimension D=256) that are learned end-to-end during training through backpropagation—they start as random noise but gradually learn to represent different "search patterns" or "detection slots": for example, 

- in the detection decoder with 100 queries, Query #1 might learn to specialize in finding large vehicles in the center-left region, Query #50 might specialize in small vehicles in the bottom-right, and Query #80 might focus on distant objects—the network automatically discovers these specializations. 

- **For Detection Decoder**: 100 queries (Q_det ∈ ℝ^(100×256)) are initialized randomly and learn to propose diverse object locations across the image; each query competes to detect one vehicle and outputs a bounding box + class. 

- **For Segmentation Decoder**: 50 queries (Q_seg ∈ ℝ^(50×256)) learn to represent different mask "prototypes"—some might specialize in horizontal driveway strips, others in curved footpath boundaries—and each outputs a binary mask. 

- **For Plate Detection Decoder**: 50 queries (Q_plate ∈ ℝ^(50×256)) learn to search for rectangular plate-like regions, guided by detection features to focus near vehicles. 

- **For OCR Decoder**: 20 queries (Q_ocr ∈ ℝ^(20×256)) are **sequential/positional**—Query #1 learns to find the first character, Query #2 the second character, etc., forming an autoregressive sequence like "D-L-0-1-A-B-1-2-3-4". 

- **For Tracking Decoder**: queries combine spatial queries with **temporal embeddings** (previous frame positions + motion vectors), learning to associate objects across time: "this car at (x₁,y₁) in frame t is the same as that car at (x₂,y₂) in frame t+1". The key insight is that queries are NOT hand-designed features—they're **learned representations** that discover optimal search strategies through the training data, allowing each decoder to specialize in its task while all sharing the same underlying encoder features.
