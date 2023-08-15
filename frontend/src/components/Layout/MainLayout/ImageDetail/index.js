import React, {useCallback, useEffect, useState} from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import {toast} from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';


function ImageDetails({match}) {
    const {image_url} = match.params;
    const [imageDetails, setImageDetails] = useState(null);


    useEffect(() => {
        // Fetch image details from the backend based on the image name
        const fetchImageDetails = () => {
            axios.get(`/api/browse/${image_url}`)
                .then(response => {
                    const {data} = response;
                    setImageDetails(data);
                })
                .catch(error => {
                    console.error(error);
                    toast.error("Image is not found");
                });
        };

        fetchImageDetails();
    }, [image_url]);

    const [selectedLabel, setSelectedLabel] = useState(imageDetails.predicted_class);

    const handleApprove = useCallback(
        (dbId, isApproved) => {

        if (selectedLabel) {

            const requestData = {
                approved: isApproved,
                label: selectedLabel
            };

            axios
                .post(`/api/update_approval/${dbId}`, requestData)
                .then((response) => {
                    // Update the predicted_class in the frontend
                    setImageDetails((prevDetails) => ({
                        ...prevDetails,
                        predicted_class: selectedLabel,
                        approved: isApproved,
                    }));
                })
                .catch((error) => {
                    console.error(error);
                });
        }
    }, [selectedLabel, imageDetails]);

    if (!imageDetails) {
        return <p>Loading...</p>;
    }

    const {
        image_name,
        predicted_class,
        approved,
        probabilities,
        magnification,
        start_day,
        start_month,
        start_year,
        plate_index,
        image_index,
        well_index,
        linker,
        reaction_time,
        temperature,
        ctot,
        loglmratio,
    } = imageDetails;


    const crystal = {
        x: [probabilities.crystal], y: [''], name: 'Crystal', orientation: 'h', marker: {
            color: 'rgb(52, 129, 237)', width: 1
        }, type: 'bar'
    };

    const challenging_crystal = {
        x: [probabilities.challenging_crystal], y: [''], name: 'Challenging Crystal', orientation: 'h', marker: {
            color: 'rgb(130, 232, 133)', width: 1
        }, type: 'bar'
    };

    const non_crystal = {
        x: [probabilities.non_crystal], y: [''], name: 'Non Crystal', orientation: 'h', marker: {
            color: 'rgb(224, 93, 70)', width: 1
        }, type: 'bar'
    };

    var data = [crystal, challenging_crystal, non_crystal];

    const layout = {
        xaxis: {range: [0, 1]}, barmode: 'stack', width: 700, // Set the desired width
        height: 80,
        margin: {l:0,r:0,b:0,t:0},
        showlegend: false,
    }

    const formattedStartDate = `Experiment date: Day: ${start_day} - Month: ${start_month} - Year: ${start_year}`;


    return (<div>
        <h2>Image Details</h2>
        <p>Image Name: {imageDetails.image_name}</p>
    <div style={{ display: 'flex' }}>
        
        {/* Left side: Image */}
        <div style={{ flex: 1, marginRight: '20px' }}>
            
            <div>
                <img
                    src={`/api/${imageDetails.image_path}`}
                    alt={imageDetails.image_name}
                    style={{maxWidth: '700px'}}
                />
            </div>
            {imageDetails.approved ? (<p style={{color: 'green'}}>
                    Predicted Class: {predicted_class} (approved)
                </p>) : (<p style={{color: 'red'}}>
                    Predicted Class: {predicted_class} (tentative)
                </p>)}

            <select
                value={selectedLabel}
                onChange={(e) => setSelectedLabel(e.target.value)}
            >
                <option value="crystal">crystal</option>
                <option value="challenging_crystal">challenging crystal</option>
                <option value="non_crystal">non crystal</option>
            </select>

            {/* Buttons */}
            <button onClick={() => handleApprove(imageDetails.db_id, true)}>Approve</button>
            <button onClick={() => handleApprove(imageDetails.db_id, false)}>Tentative</button>
        </div>

         {/* Right side: Details */}
         <div style={{ flex: 1 }}>
            <div>
                <h3>Probabilities</h3>
                <ul>
                    <li>Crystal: {probabilities.crystal.toFixed(3)}</li>
                    <li>Challenging Crystal: {probabilities.challenging_crystal.toFixed(3)}</li>
                    <li>Non Crystal: {probabilities.non_crystal.toFixed(3)}</li>
                </ul>
            </div>
            
            <div>
                <Plot data={data} layout={layout}/>
            </div>

            {/* Display other image details */}
            <p>Magnification: {magnification}</p>
            <p>{formattedStartDate}</p>
            <p>Plate Index: {plate_index}</p>
            <p>Image Index: {image_index}</p>
            <p>Well Index: {well_index}</p>
            <p>Linker: {linker}</p>
            <p>Reaction Time: {reaction_time}</p>
            <p>Temperature: {temperature}</p>
            <p>Ctot: {ctot}</p>
            <p>Loglmratio: {loglmratio}</p>
        </div>
        </div>
        </div>);
};

export default ImageDetails
