import React, {useEffect, useState} from 'react';
import axios from 'axios';
import {useHistory} from "react-router-dom";

export default function FilterBar() {
    const [labelOptions, setLabelOptions] = useState([]);
    const [statusOptions, setStatusOptions] = useState([]);
    const [linkerOptions, setLinkerOptions] = useState([]);
    const [magnificationOptions, setMagnificationOptions] = useState([]);
    const [reactionTimeOptions, setReactionTimeOptions] = useState([]);
    const [temperatureOptions, setTemperatureOptions] = useState([]);

    const [label, setLabel] = useState();
    const [status, setStatus] = useState();
    const [linker, setLinker] = useState();
    const [magnification, setMagnification] = useState();
    const [reactionTime, setReactionTime] = useState();
    const [temperature, setTemperature] = useState();

    useEffect(() => {
        fetchFilterOptions();
    }, []);

    const fetchFilterOptions = async () => {
        try {
            const response = await axios.get('/api/filter_unique_values');
            const data = response ? response.data : null;

            console.log(data)
            setLabelOptions(data.labelOptions);
            setStatusOptions(data.statusOptions);
            setLinkerOptions(data.linkerOptions);
            setMagnificationOptions(data.magnificationOptions);
            setReactionTimeOptions(data.reactionTimeOptions);
            setTemperatureOptions(data.temperatureOptions);
        } catch (error) {
            console.error(error);
        }
    };

    const history = useHistory()

    const handleApplyFilter = () => {
        const queryParams = new URLSearchParams();
        if (label) {
            queryParams.append('assigned_label', label);
        }
        if (status) {
            console.log('Status:',status)
            queryParams.append('approved', status==='true');
        }
        if (linker) {
            queryParams.append('linker', linker);
        }
        if (magnification) {
            queryParams.append('magnification', parseInt(magnification));
        }
        if (reactionTime) {
            queryParams.append('reaction_time', parseInt(reactionTime));
        }
        if (temperature) {
            queryParams.append('temperature', parseInt(temperature));
        }

        console.log(queryParams)
        history.push(`/browse?${queryParams.toString()}`);
    };

    return (
        <div>
            <h3>Filter Options</h3>
            <div className="filter-bar">
                <div className="filter-field">
                    <label htmlFor="label">Label: </label>
                    <select id="label" value={label} onChange={(e) => setLabel(e.target.value)}>
                        <option value="">All</option>
                        {labelOptions ? labelOptions.map((option, index) => (
                            <option key={index} value={option}>
                                {option}
                            </option>
                        )) : null}
                    </select>
                </div>
                <div className="filter-field">
                    <label htmlFor="status">Label Status: </label>
                    <select id="status" value={status} onChange={(e) => setStatus(e.target.value)}>
                        <option value="">All</option>
                        {statusOptions ? statusOptions.map((option, index) => (
                            <option key={index} value={option}>
                                {option ? "Approved" : "Tentative"}
                            </option>
                        )) : null}
                    </select>
                </div>
                <div className="filter-field">
                    <label htmlFor="linker">Linker: </label>
                    <select id="linker" value={linker} onChange={(e) => setLinker(e.target.value)}>
                        <option value="">All</option>
                            {linkerOptions ? linkerOptions.map((option, index) => (
                                <option key={index} value={option}>
                                    {option}
                                </option>
                            )) : null}
                    </select>
                </div>
                <div className="filter-field">
                    <label htmlFor="magnification">Magnification: </label>
                    <select id="magnification" value={magnification} onChange={(e) => setMagnification(e.target.value)}>
                        <option value="">All</option>
                            {magnificationOptions ? magnificationOptions.map((option, index) => (
                                <option key={index} value={option}>{option}</option>
                            )) : null}
                    </select>
                </div>
                <div className="filter-field">
                    <label htmlFor="reactionTime">Reaction time: </label>
                    <select id="reactionTime" value={reactionTime} onChange={(e) => setReactionTime(e.target.value)}>
                        <option value="">All</option>
                            {reactionTimeOptions ? reactionTimeOptions.map((option, index) => (
                                <option key={index} value={option}>{option}</option>
                            )) : null}
                    </select>
                </div>
                <div className="filter-field">
                    <label htmlFor="temperature">Temperature: </label>
                    <select id="temperature" value={temperature} onChange={(e) => setTemperature(e.target.value)}>
                        <option value="">All</option>
                            {temperatureOptions ? temperatureOptions.map((option, index) => (
                                <option key={index} value={option}>{option}</option>
                            )) : null}
                    </select>
                </div>
            </div>

            <button onClick={handleApplyFilter}>Apply</button>
        </div>
    );
};
