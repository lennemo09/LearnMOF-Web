import React, {useState} from 'react';
import axios from 'axios';
import {useHistory} from 'react-router-dom';
import {toast, ToastContainer} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function FileUpload() {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [selectedMetadata, setSelectedMetadata] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
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
                const uploadResponse = await axios.post('/api/upload', formData);
                console.log(uploadResponse.data); // Handle the response from the backend
                setUploadSuccess(true);

                const image_paths = uploadResponse.data.image_paths;

                const processResponse = await axios.post('/api/process_images', {
                    image_paths,
                });
                console.log(processResponse)
                const image_ids = processResponse.data
                const queryParams = image_ids.map((image_id) => `image_ids=${image_id}`).join('&');
                history.push(`/browse?${queryParams}`);
            } catch
                (error) {
                console.error(error.response.data); // Handle error if any
                toast.error(error.response.data); // Display toaster notification for error
            }
        }
    ;


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
            {uploadSuccess && <p>Files uploaded successfully</p>}
            <ToastContainer/> {/* Toaster container for displaying notifications */}
        </div>
    );
}

export default FileUpload;
