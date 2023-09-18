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
        console.log(params)
        axios
            .get('/api/browse', {params: params, 
                                paramsSerializer: {indexes: null}})
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










