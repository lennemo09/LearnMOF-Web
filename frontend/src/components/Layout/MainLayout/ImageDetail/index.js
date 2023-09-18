import React, {useCallback, useEffect, useState} from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import {useLocation, useHistory} from "react-router-dom";
import {toast} from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';


function ImageDetails({match}) {
    const {image_url} = match.params;
    const [imageDetails, setImageDetails] = useState(null);
    const [previousImageId, setPreviousImageId] = useState(null); // Define the state
    const [nextImageId, setNextImageId] = useState(null); // Define the state
    const [selectedLabel, setSelectedLabel] = useState('crystal');

    const history = useHistory();

    useEffect(() => {
        // Fetch image details from the backend based on the image name
        const fetchImageDetails = () => {
            axios.get(`/api/browse/${image_url}`)
                .then(response => {
                    const {data} = response;
                    setImageDetails(data);
                    setSelectedLabel(data.predicted_class);
                })
                .catch(error => {
                    console.error(error);
                    toast.error("Image is not found");
                });
        };

        fetchImageDetails();

        // Fetch all image IDs from Flask backend
        axios.get('/api/all_image_ids')
        .then(response => {
            const imageIds = response.data;
            const currentIndex = imageIds.findIndex((id) => id === image_url);
            console.log(imageIds);
            console.log(currentIndex);
            // Determine IDs of images before and after
            const previousId = currentIndex > 0 ? imageIds[currentIndex - 1] : imageIds[imageIds.length - 1];
            const nextId = currentIndex < imageIds.length - 1 ? imageIds[currentIndex + 1] : imageIds[0] ;

            // Use previousImageId and nextImageId as needed
            setPreviousImageId(previousId); // Update the state
            setNextImageId(nextId); // Update the state
            console.log(previousImageId, nextImageId)
        })
        .catch(error => {
            console.error(error);
        });

    }, [image_url]);

    

    const goToPreviousImage = () => {
        if (previousImageId) {
            history.push(`/browse/${previousImageId}`);
        }
    };

    const goToNextImage = () => {
        if (nextImageId) {
            history.push(`/browse/${nextImageId}`);
        }
    };

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

    <div style={{ display: 'flex' }}>
        <div style={{ flex: 1, marginRight: '20px' }}>
            <h2>Image Details</h2>
            <p>Image Name: {imageDetails.image_name}</p>
        </div>
        
        {/* Next and Previous buttons */}
        <div style={{ flex: 1 }}> 
        <br/>
        <br/>
            <button className="nav-button" onClick={goToPreviousImage} disabled={!previousImageId}>
                ❮
            </button>
            <span style={{margin: '0 10px'}}></span>
            <button className="nav-button" onClick={goToNextImage} disabled={!nextImageId}>
                ❯
            </button>
        </div>
    </div>
    
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
                defaultValue={selectedLabel}
            >
                <option value="crystal">crystal</option>
                <option value="challenging_crystal">challenging crystal</option>
                <option value="non_crystal">non crystal</option>
            </select>

            {/* Buttons */}
            <span style={{margin: '0 10px'}}>|</span>
            <button className="approve-button" onClick={() => handleApprove(imageDetails.db_id, true)}>Approve</button>
            <span style={{margin: '0 10px'}}></span>
            <button className="tentative-button" onClick={() => handleApprove(imageDetails.db_id, false)}>Tentative</button>

            <br/>
            <br/>

            
        </div>

         {/* Right side: Details */}
         <div style={{ flex: 1 }}>
            <div>
                <h3>Probabilities</h3>
                <ul>
                    <li>Crystal: {probabilities.crystal.toFixed(3)}</li>
                    <li>Challenging crystal: {probabilities.challenging_crystal.toFixed(3)}</li>
                    <li>Non-crystal: {probabilities.non_crystal.toFixed(3)}</li>
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
