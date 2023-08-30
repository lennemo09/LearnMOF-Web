import React, {useState, useEffect} from 'react';
import axios from 'axios';
import {useHistory} from 'react-router-dom';
import {toast, ToastContainer} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function FileUpload() {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [selectedMetadata, setSelectedMetadata] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [preparingSuccess, setPreparingSuccess] = useState(false);
    const [inferenceSuccess, setInferenceSuccess] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [preparingProgress, setPreparingProgress] = useState(0);
    const [inferenceProgress, setInferenceProgress] = useState(0);
    const [processId, setProcessId] = useState(null);

    const history = useHistory();

    const handleFileUpload = (event) => {
        const files = event.target.files;
        setSelectedFiles([...files]);
    };

    const handleMetadataUpload = (event) => {
        const file = event.target.files[0];
        setSelectedMetadata(file);
    };

    const handleSubmit = async () => {
            const formData = new FormData();

            selectedFiles.forEach((file) => {
                formData.append('images', file);
            });

            if (selectedMetadata) {
                formData.append('metadata', selectedMetadata);
            }

            try {
                const response = await axios.get(`/api/get_inference_process_id`);
                // Set the processId state with the generated processId
                setProcessId(response.data.process_id);
                formData.append('process_id', response.data.process_id);
            } catch (error) {
                console.error('Error fetching inference process id:', error);
            }

            try {
                // Write toast notification saying "Uploading images..."
                toast.info("Uploading images...");
                const uploadResponse = await axios.post('/api/upload', formData, {
                    onUploadProgress: (progressEvent) => {
                        const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                        // Update your progress indicator here, e.g., setUploadProgress(progress);
                        // console.log(`Upload Progress: ${progress}%`);
                        setUploadProgress(progress);
                    }
                });
                console.log(uploadResponse.data); // Handle the response from the backend
                setUploadSuccess(true);
                setUploadProgress(100);
                toast.info("Upload succeeded. Processing images...");

                const image_paths = uploadResponse.data.image_paths;

                const processResponse = await axios.post('/api/process_images', {
                    image_paths, processId,
                });
                console.log(processResponse)
                setPreparingSuccess(true);
                
                const image_ids = processResponse.data.image_ids
                const queryParams = image_ids.map((image_id) => `image_ids=${image_id}`).join('&');
                
                setInferenceSuccess(true);

                history.push(`/browse?${queryParams}`);
            } catch
                (error) {
                console.error(error.response.data); // Handle error if any
                toast.error(error.response.data); // Display toaster notification for error
            }
        }
    ;
    
    useEffect(() => {
        // Poll the inference progress every 1 seconds
        const interval = setInterval(async () => {
            
            try {
                const response = await axios.get(`api/prepare_data_progress/${processId}`);
                setPreparingProgress(response.data.progress);
                // console.log('Data prep progress:', response.data.progress)
            } catch (error) {
                console.error('Error fetching data preparation progress:', error);
            }
        }, 1000);

        return () => {
            // setPreparingProgress(100);
            clearInterval(interval); // Clear interval when component unmounts
        };
    }, [processId]); // Make sure to include processId as a dependency

    useEffect(() => {
        // Poll the inference progress every 1 seconds
        const interval = setInterval(async () => {
            try {
                const response = await axios.get(`api/inference_progress/${processId}`);
                setInferenceProgress(response.data.progress);
                // console.log('Inference progress:', response.data.progress)
            } catch (error) {
                console.error('Error fetching inference progress:', error);
            }
        }, 1000);

        return () => {
            // setInferenceProgress(100);
            clearInterval(interval); // Clear interval when component unmounts
        };
    }, [processId]); // Make sure to include processId as a dependency

    return (
        <div>
            <div>
                <label htmlFor="images">Select Images:</label>
                <input type="file" id="images" multiple onChange={handleFileUpload}/>
            </div>
            <div>
                <label htmlFor="metadata">Select Metadata (CSV):</label>
                <input type="file" id="metadata" accept=".csv" onChange={handleMetadataUpload}/>
            </div>
            <div>
                <h3>Selected Files:</h3>
                <ul>
                    {selectedFiles.map((file, index) => (
                        <li key={index}>{file.name}</li>
                    ))}
                </ul>
            </div>

            <div>
                <h3>Selected Metadata File:</h3>
                <ul>
                    {selectedMetadata && <li>{selectedMetadata.name}</li>}
                </ul>
            </div>

            <button onClick={handleSubmit}>Upload</button>
            {uploadProgress > 0 && <p>Uploading images... {Math.floor(uploadProgress)}%</p>}
            {preparingProgress > 0 && <p>Preparing images... {Math.floor(preparingProgress)}%</p>}
            {inferenceProgress > 0 && <p>Processing images... {Math.floor(inferenceProgress)}%</p>}
            {uploadSuccess && <p>Files uploaded successfully. Waiting for images to be processed...</p>}
            {inferenceSuccess && <p>Images processed successfully. Redirecting to database view...</p>}
            <ToastContainer/> {/* Toaster container for displaying notifications */}
        </div>
    );
}

export default FileUpload;
