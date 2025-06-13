# Final Napari Group Project - SVS File Processing Plugins

**Team:** Vish-AJ-Ivan

This project contains four complementary napari plugins designed for processing and analyzing whole slide images (WSI) in SVS format. Each plugin addresses different aspects of SVS file handling, from basic loading to advanced nuclear segmentation.

## Installation:
Clone the repository and install the provided environment.yml using conda or pixi.
Then, install the plugin of interest using

```
pip install ./<plugin folder>
```
Then, simply run Napari, activate the plugin, and import an SVS file.

## Plugin Overview

### 1. **preload-svs-reader**
**Purpose:** Siimple plugin for eagerly loading and displaying .svs files in Napari

**Use Case:** 
- When you need fast, immediate access to small SVS files
- Suitable for smaller files or when memory usage is not a concern
- Best for interactive exploration where responsiveness is prioritized over memory efficiency
- VERY SLOW to load with large .svs files

**Design Considerations:**
- Eager loading strategy loads entire image data into memory upfront
- Provides immediate responsiveness for panning and zooming once file is loaded
- Trade-off: Higher memory usage for better performance

### 2. **lazy-svs-loader** 
**Purpose:** Plugin to lazily load .svs data files for use in Napari

**Use Case:**
- Large SVS files where memory efficiency is critical
- Can load large files quickly
- Suitable for basic viewing and navigation of large whole slide images

**Design Considerations:**
- Lazy loading strategy loads data on-demand
- Memory efficient - only loads visible portions of the image
- Optimized for handling very large files that wouldn't fit in memory
- May have slight latency during navigation as data is loaded dynamically

### 3. **lazy-loader-hed**
**Purpose:** Lazy loader with H&E deconvolution capabilities

**Use Case:**
- Histopathology workflows requiring separation of Hematoxylin and Eosin stain channels
- Color deconvolution analysis of H&E stained tissue sections

**Design Considerations:**
- Combines lazy loading efficiency with specialized H&E processing
- Implements color deconvolution algorithms for stain separation

### 4. **stardist-svs-nuclear-segmenter**
**Purpose:** Uses StarDist deep learning model to perform nuclear segmentation on .svs images

**Use Case:**
- Automated nuclear detection and segmentation in histopathology images

**Design Considerations:**
- Integrates StarDist deep learning model for accurate nuclear segmentation
- Provides both image viewing and segmentation layers in napari
- Includes both widget-based manual segmentation and automatic reader-based processing
- Handles multiscale pyramids with segmentation applied to highest resolution level
- Uses Dask arrays for efficient memory management during segmentation

## Architecture & Design Philosophy

### Multiscale Image Pyramid Support
All plugins leverage OpenSlide's DeepZoom functionality to create multiscale pyramids, enabling efficient navigation across different zoom levels typical in whole slide imaging.

### Dask Integration
The plugins use Dask arrays for:
- Lazy computation and memory efficiency
- Parallel processing capabilities
- Integration with napari's multiscale viewing

## Technical Stack

- **Core:** napari, OpenSlide, Dask
- **Image Processing:** NumPy, StarDist (for segmentation)
- **UI:** magicgui for widget components
- **File Format:** SVS (Aperio whole slide images)

## Use Case Matrix

| Plugin | Memory Usage | Loading Speed | Processing Speed | Specialized Features | Best For |
|--------|--------------|---------------|------------------|---------------------|----------|
| preload-svs-reader | High | Very Slow | Fast | Basic viewing | Small files, interactive use |
| lazy-svs-loader | Low | Fast | Moderate | Efficient loading | Large files, basic viewing |
| lazy-loader-hed | Low | Fast | Moderate | H&E deconvolution | Histopathology color analysis |
| stardist-svs-nuclear-segmenter | Moderate | Fast | Variable | AI-powered segmentation | Automated nuclear analysis |