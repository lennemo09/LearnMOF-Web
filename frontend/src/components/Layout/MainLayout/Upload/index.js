import React, {useState, useEffect} from 'react';
import axios from 'axios';
import {useHistory} from 'react-router-dom';
import {toast, ToastContainer} from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

function FileUpload() {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [selectedMetadata, setSelectedMetadata] = useState(null);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [sessionToken, setSessionToken] = useState(null);
    const [preparingSuccess, setPreparingSuccess] = useState(false);
    const [inferenceSuccess, setInferenceSuccess] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [preparingProgress, setPreparingProgress] = useState(0);
    const [inferenceProgress, setInferenceProgress] = useState(0);
    const [processId, setProcessId] = useState(null);

    let newSessionToken;

    const history = useHistory();

    const handleFileUpload = (event) => {
        const files = event.target.files;
        setSelectedFiles([...files]);
    };

    const handleMetadataUpload = (event) => {
        const file = event.target.files[0];
        setSelectedMetadata(file);
    };

    const getCookie = (name) => {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    };

    // Function to generate a new session token (UUID)
    const generateSessionToken = () => {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
        });
    };

    useEffect (() => {
        // Check if the session token exists in the cookie
        const existingSessionToken = getCookie('session_token');
    
        if (!existingSessionToken) {
            // If it doesn't exist, generate a new UUID
            try {
                const newSessionTokenResponse = axios.get(`/api/get_session_token`);
                // Set the processId state with the generated processId
                newSessionToken = newSessionTokenResponse.data.session_token;
                setSessionToken(newSessionTokenResponse.data.session_token);
            } catch (error) {
                console.error('Error fetching session token from server:', error);
                // If it doesn't exist, generate a new UUID
                newSessionToken = generateSessionToken();
                setSessionToken(newSessionToken);
            }
    
            // Set the session token in a cookie (with an expiration time, e.g., 1 day)
            const expirationDate = new Date();
            expirationDate.setDate(expirationDate.getDate() + 1); // 1 day
            document.cookie = `session_token=${newSessionToken}; expires=${expirationDate.toUTCString()}`;
    
            // Set the session token in the state
            setSessionToken(newSessionToken);
        } else {
            newSessionToken = existingSessionToken;
        }
        console.log("Initialized session token:", newSessionToken)
    }, []);
    
    const handleSubmit = async () => {
            const formData = new FormData();
            const cookiedToken = getCookie('session_token');

            selectedFiles.forEach((file) => {
                formData.append('images', file);
            });

            if (selectedMetadata) {
                formData.append('metadata', selectedMetadata);
            }

            if (cookiedToken) {
                formData.append('process_id', cookiedToken);
            }
            console.log('Using session token:', cookiedToken)

            try {
                // Write toast notification saying "Uploading images..."
                toast.info("Uploading images...");
                const uploadResponse = await axios.post('/api/upload', formData, {
                    timeout: 6000000,
                    onUploadProgress: (progressEvent) => {
                        const progress = Math.round((progressEvent.loaded / progressEvent.total) * 100);
                        // Update your progress indicator here, e.g., setUploadProgress(progress);
                        // console.log(`Upload Progress: ${progress}%`);
                        setUploadProgress(progress);
                        setUploadSuccess(false);
                    }
                }).then((response) => {setUploadSuccess(true)});

                // if (uploadProgress.data)
                // {
                //     setUploadSuccess(true);
                //     setUploadProgress(100);
                //     toast.info("Upload succeeded. Processing images...");
                //     const image_paths = uploadResponse.data.image_paths;

                //     const processResponse = await axios.post('/api/process_images', {
                //         timeout: 6000000,
                //         image_paths, processId,
                //     });

                //     if (processResponse.data) {
                //         console.log(processResponse)
                //         setPreparingSuccess(true);
                        
                //         const image_ids = processResponse.data.image_ids
                //         const queryParams = image_ids.map((image_id) => `image_ids=${image_id}`).join('&');
                        
                //         setInferenceSuccess(true);
        
                //         history.push(`/browse?${queryParams}`);
                //     }
                // }


            } catch
                (error) {
                console.error(error); // Handle error if any
                toast.error(error); // Display toaster notification for error
            }
        }
    ;
    
    useEffect(() => {
        // Poll the inference progress every 1 seconds
        const interval = setInterval(async () => {
            
            try {
                const response = await axios.get(`api/prepare_data_progress/${getCookie('session_token')}`);
                setPreparingProgress(response.data.progress);

                const progress = response.data.progress;
                if (progress < 0)  { setPreparingSuccess(true) } else {setPreparingSuccess(false); setPreparingProgress(response.data.progress);}
                
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
                const response = await axios.get(`api/inference_progress/${getCookie('session_token')}`);

                const progress = response.data.progress;
                if (progress < 0)  { setInferenceSuccess(true) } else {setInferenceSuccess(false); setInferenceProgress(response.data.progress);}
                
                // console.log('Inference progress:', response.data.progress)
            } catch (error) {
                console.error('Error fetching inference progress:', error);
            }
        }, 1000);

        return () => {
            // setInferenceProgress(100);
            setInferenceSuccess(true)
            clearInterval(interval); // Clear interval when component unmounts
        };
    }, [processId]); // Make sure to include processId as a dependency

    return (
        <div>
            <div>
                <label htmlFor="images">Select Images: </label>
                <input type="file" id="images" multiple onChange={handleFileUpload}/>
            </div>
            <div>
                <label htmlFor="metadata">Select Metadata (CSV): </label>
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
            {<p>Progress for session token: {getCookie('session_token')}</p>}

            {uploadProgress > 0 && !uploadSuccess &&<p>Uploading images... {uploadProgress.toFixed(1)}%</p>}
            {uploadSuccess && <p>No ongoing upload processes for this session token.</p>}
            {preparingProgress > 0 && !preparingSuccess && <p>Processing images on the server: {preparingProgress.toFixed(1)}%. You can safely close this page.</p>}
            {inferenceProgress > 0 && !inferenceSuccess && <p>Running inference model: {inferenceProgress.toFixed(1)}%. You can safely close this page.</p>}
            {inferenceSuccess && <p>No ongoing inference tasks for this session token. If you've recently uploaded images, please head to Browse section.</p>}
            <ToastContainer/> {/* Toaster container for displaying notifications */}
        </div>
    );
}

export default FileUpload;
