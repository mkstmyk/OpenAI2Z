# OpenAI to Z Challenge - Checkpoint 1 & 2

A remote sensing tool for archaeology that downloads Sentinel-2 L2A COG (GeoTIFF) data for the Amazon basin, automatically extracts NDVI anomaly regions, and generates archaeological explanations using OpenAI GPT-4o-mini.

## 🚀 Features

### ✅ Implemented Features
- **Sentinel-2 Data Download**: Automatic retrieval from AWS S3
- **NDVI Calculation**: Vegetation index computation and visualization
- **Anomaly Extraction**: Automatic detection based on thresholding
- **OpenAI Analysis**: Archaeological interpretation using GPT-4o-mini
- **Checkpoint 1**: Multi-source data ingestion
- **Checkpoint 2**: New site discovery (algorithmic detection + historical text + known site comparison)

### 🔧 Debug Features
- **Skip OpenAI API**: Avoids credit consumption during debugging
- **Dummy Data Generation**: Alternative when real data is unavailable
- **Stepwise Execution**: Option to skip heavy processing

## 📊 Data Source Status

### ✅ Working Data Sources
- **Sentinel-2**: Direct download from AWS S3
- **Sample Archaeological Data**: Locally generated (as a substitute for real data)
- **Sample Elevation Data**: Locally generated (as a substitute for SRTM)
- **Vegetation Data**: Derived from Sentinel-2 (as a substitute for GEDI)

### ⚠️ Limitations
- **TerraBrasilis**: URL resolution error (using alternative data)
- **OpenTopography SRTM**: API 404 error (using alternative data)
- **GEDI L2A**: Direct access difficult (using Sentinel-2 derived data)

### 🔄 Planned Improvements
- **NASA Earthdata API**: More reliable SRTM data
- **OpenStreetMap**: Public data for archaeological sites
- **UNESCO**: World Heritage site data

## 🛠️ Setup

### 1. Prepare Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
```bash
# Set environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. AWS CLI Setup (for Sentinel-2 data)
```bash
# Install AWS CLI (if not installed)
pip install awscli

# Configure (no credentials needed - public data)
aws configure set default.s3.signature_version s3v4
```

## 🚀 Usage

### Basic Execution
```bash
python openai_to_z_checkpoint.py
```

### Debug Mode (Recommended)
```python
# In main() of openai_to_z_checkpoint.py
DEBUG_MODE = True  # Skip OpenAI API
```

### Production Mode
```python
# In main() of openai_to_z_checkpoint.py
DEBUG_MODE = False  # Use actual OpenAI API calls
```

## 📁 Output Files

### Main Analysis
- `data_dir/footprints.json`: Detected anomaly regions
- `data_dir/ndvi_map.png`: NDVI visualization
- `openai_log.json`: OpenAI analysis log

### Checkpoint 1
- `data_dir/archaeological_sites.geojson`: Archaeological site data
- `data_dir/srtm_elevation.tif`: Elevation data
- `data_dir/vegetation_data.json`: Vegetation data

### Checkpoint 2
- `data_dir/checkpoint2_candidates.geojson`: Algorithmic detection results
- `data_dir/historical_extracts.json`: Extracted historical texts
- `data_dir/site_comparison.json`: Known site comparison results

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Error**
   ```bash
   # Check environment variable
   echo $OPENAI_API_KEY

   # Test in debug mode
   DEBUG_MODE = True
   ```

2. **Sentinel-2 Download Error**
   ```bash
   # Check AWS CLI
   aws --version

   # Check network connection
   curl -I https://sentinel-s2-l2a.s3.amazonaws.com
   ```

3. **Out of Memory Error**
   ```python
   # Skip heavy processing
   skip_heavy = True
   ```

### Data Source Issues

1. **If Real Data Is Unavailable**
   - Sample data is generated automatically
   - Processing continues and feature testing is possible

2. **If Specific Data Sources Are Needed**
   - Download data manually
   - Place in `data_dir/`
   - Re-run the script

## 📈 Performance

### Typical Runtime
- **Debug Mode**: 2-3 minutes
- **Production Mode**: 5-10 minutes (including API calls)
- **Skip Heavy Processing**: 1-2 minutes

### Memory Usage
- **Basic Processing**: 500MB-1GB
- **Heavy Processing**: 2-4GB
- **Recommended**: 8GB or more

## 🎯 Checkpoint Requirements Status

### Checkpoint 1 ✅
- [x] Ingest multiple independent data sources
- [x] Generate 5+ anomaly footprints
- [x] Log dataset IDs and OpenAI prompts
- [x] Reproducible script

### Checkpoint 2 ✅
- [x] Algorithmic detection (Hough transform)
- [x] Historical text extraction (using GPT)
- [x] Comparison with known archaeological sites

## 🤝 Contributing

1. Add new real data sources
2. Improve algorithms
3. Strengthen error handling
4. Improve documentation

## 📄 License

MIT License - Open source project

## 🔗 References

- [OpenAI to Z Challenge](https://openai.com/blog/openai-to-z-challenge)
- [Starter Pack](documents/starter-pack-openai-to-z-challenge.txt)
- [Checkpoints Guide](documents/checkpoints-openai-to-z-challenge.txt) 