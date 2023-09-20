import React, {useEffect, useState} from 'react';
import axios from 'axios';
import {useLocation, useHistory} from "react-router-dom";
import FilterBar from "../FilterBar";

import {toast, ToastContainer} from 'react-toastify';

export default function Browse() {
    const [images, setImages] = useState([]);

    const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false); // Step 2
    const [imageToDelete, setImageToDelete] = useState(null);

    const history = useHistory();

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const image_ids = queryParams.getAll('image_ids');


    const approved = queryParams.get("approved");
    const assignedLabel = queryParams.get("assigned_label");
    const linker = queryParams.get("linker");
    const magnification = queryParams.get("magnification");
    const reactionTime = queryParams.get("reaction_time");
    const temperature = queryParams.get("temperature");
    const year = queryParams.get("year");
    const month = queryParams.get("month");
    const day = queryParams.get("day");
    
    useEffect(() => {
        fetchImages();
    }, [location.search]);
    
    
    const fetchImages = () => {
        const params = {
            approved: approved, 
            image_ids: image_ids, 
            assigned_label:assignedLabel,
            linker:linker,
            magnification:magnification ? parseInt(magnification) : null,
            reaction_time:reactionTime ? parseInt(reactionTime) : null,
            temperature:temperature ? parseInt(temperature) : null,
            start_date_year: year ? parseInt(year) : null,
            start_date_month: month ? parseInt(month) : null,
            start_date_day: day ? parseInt(day) : null,
        }
        axios
            .get('/api/browse', {params: params, 
                                paramsSerializer: {indexes: null}})
            .then((response) => {
                setImages(response.data);
            })
            .catch((error) => {
                console.error(error);
                toast.error(error.response.data);
            });
    };

    const handleClickImage = (db_id) => {
        history.push(`/browse/${db_id}`);
    };

    const handleDeleteImage = (db_id) => {
        // Call the API to delete the image here
        axios
          .delete(`/api/remove_image/${db_id}`)
          .then((response) => {
            // Handle success, e.g., remove the image from the state
            const updatedImages = images.filter((image) => image.db_id !== db_id);
            setImages(updatedImages);
            toast.success('Image deleted successfully.');
          })
          .catch((error) => {
            console.error(error);
            toast.error('Error deleting image.');
          });
      };

    const handleApproveImage = (db_id, assigned_label, approved) => {
        const requestData = {
            approved: approved,
            label: assigned_label
        };

        axios
            .post(`/api/update_approval/${db_id}`, requestData)
            .then((response) => {
                const updatedImages = images.map((image) => {
                    if (image.db_id == db_id) {
                        image.approved = approved;
                    }
                    return image;
                });
                setImages(updatedImages);
                toast.success('Image approval updated successfully.');
            })
            .catch((error) => {
                console.error(error);
            });
    }

    const handleShowDeleteConfirmation = (db_id) => {
    setImageToDelete(db_id);
    setShowDeleteConfirmation(true);
    };

    const handleConfirmDelete = () => {
    if (imageToDelete) {
        handleDeleteImage(imageToDelete);
        setImageToDelete(null);
    }
    setShowDeleteConfirmation(false);
    };

    const handleCancelDelete = () => {
    setImageToDelete(null);
    setShowDeleteConfirmation(false);
    };

    return (
        <div>
            <h1 style={{marginTop: '20px', textAlign: 'center'}}>Database</h1>
            <FilterBar/>
            <div className='images-count'> <p>Found {images.length} images matching filter.</p> </div>
            <div className='browse-content'>
                {images.length > 0 ? (
                        images.map((image) => (
                                <div
                                    key={image.db_id}
                                    style={{
                                        width: '300px',
                                        margin: '30px',
                                        boxShadow: '0 0 5px rgba(0, 0, 0, 0.3)',
                                        borderRadius: '5px',
                                        overflow: 'hidden',
                                    }}
                                >
                                    <div onClick={() => handleClickImage(image.db_id)}>
                                        <img
                                            src={`/api/${image.image_path}`}
                                            alt={image.image_name}
                                            style={{width: '100%'}}
                                        />
                                        <div style={{padding: '10px'}}>
                                            <h5 style={{marginBottom: '5px'}}>{image.image_name}</h5>
                                            <p style={{marginBottom: '5px'}}>Assigned Label: {image.assigned_label}</p>
                                            <p
                                                style={{
                                                    marginBottom: '0',
                                                    color: image.approved ? 'green' : 'red',
                                                }}
                                            >
                                                Approved: {image.approved ? 'Yes' : 'No'}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="browse-buttons-container"> 
                                        <span className='approval-buttons-container'>
                                            <button
                                                className="approve-button-small"
                                                onClick={() => handleApproveImage(image.db_id, image.assigned_label, true)} // Step 4
                                            >
                                                Approve
                                            </button>
                                            
                                            <button
                                                className="tentative-button-small"
                                                onClick={() => handleApproveImage(image.db_id, image.assigned_label, false)} // Step 4
                                            >
                                                Tentative
                                            </button>
                                        </span>
                                        
                                        <span className='browse-buttons-filler'></span>

                                        <span className='delete-button-container'> </span>
                                        <button
                                            className="delete-button"
                                            onClick={() => handleShowDeleteConfirmation(image.db_id)} // Step 4
                                        >
                                            Delete
                                        </button>
                                    </div>
                                </div>
                            )
                        )
                    ) :
                    (
                        <p>No images found.</p>
                    )
                }
            </div>
            <ToastContainer/>

            {/* Confirmation popup */}
            {showDeleteConfirmation && (
                <div className="delete-confirmation-overlay">
                    <div className='delete-confirmation-content'>
                        <p>Are you sure you want to delete this image?</p>
                        <button onClick={handleConfirmDelete}>Yes</button>
                        <button onClick={handleCancelDelete}>No</button>
                    </div>
                </div>
            )}

        </div>
    )
        ;
};










