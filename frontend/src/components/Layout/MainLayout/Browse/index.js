import React, {useEffect, useState} from 'react';
import axios from 'axios';
import {useLocation, useHistory} from "react-router-dom";
import FilterBar from "../FilterBar";

import {toast, ToastContainer} from 'react-toastify';

export default function Browse() {
    const [images, setImages] = useState([]);
    const history = useHistory();

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const image_ids = queryParams.getAll('image_ids');


    console.log(image_ids)
    const approved = queryParams.get("approved")
    const assignedLabel = queryParams.get("assigned_label")

    useEffect(() => {
        fetchImages();
    }, [location.search]);


    const fetchImages = () => {
        axios
            .get('/api/browse', {params: {approved: approved, image_ids: image_ids, assigned_label:assignedLabel}, paramsSerializer: {indexes: null}})
            .then((response) => {
                setImages(response.data);
                console.log(response.data)
            })
            .catch((error) => {
                console.error(error);
                toast.error(error.response.data);
            });
    };

    const handleClickImage = (db_id) => {
        history.push(`/browse/${db_id}`);
    };

    return (
        <div>
            <h1 style={{marginTop: '20px', textAlign: 'center'}}>Database</h1>
            <FilterBar/>
            <div
                style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    marginTop: '20px',
                    margin: '20px'
                }}
            >
                {images.length > 0 ? (
                        images.map((image) => (
                                <div
                                    key={image.db_id}
                                    style={{
                                        width: '300px',
                                        margin: '10px',
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
                                    </div>
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
                            )
                        )
                    ) :
                    (
                        <p>No images found.</p>
                    )
                }
            </div>
            <ToastContainer/>
        </div>
    )
        ;
};










