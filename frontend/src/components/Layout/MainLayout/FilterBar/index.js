import React, {useState} from 'react';
import axios from 'axios';
import {useHistory} from "react-router-dom";

export default function FilterBar() {
    const [label, setLabel] = useState();
    const [labelStatus, setLabelStatus] = useState();
    const history = useHistory()

    const handleApplyFilter = () => {
        const queryParams = new URLSearchParams();
        if (label) {
            queryParams.append('assigned_label', label);
        }
        if (labelStatus) {
            queryParams.append('approved', labelStatus==='approved');
        }
        console.log(queryParams)
        history.push(`/browse?${queryParams.toString()}`);
    };

    return (
        <div>
            <h3>Filter Options</h3>
            <div>
                <label htmlFor="label">Label:</label>
                <select id="label" value={label} onChange={(e) => setLabel(e.target.value)}>
                    <option value="">All</option>
                    <option value="crystal">Crystal</option>
                    <option value="challenging_crystal">Challenging Crystal</option>
                    <option value="non_crystal">Non Crystal</option>
                </select>
            </div>
            <div>
                <label htmlFor="labelStatus">Label Status:</label>
                <select id="labelStatus" value={labelStatus} onChange={(e) => setLabelStatus(e.target.value)}>
                    <option value="">All</option>
                    <option value="approved">Approved</option>
                    <option value="tentative">Tentative</option>
                </select>
            </div>
            <button onClick={handleApplyFilter}>Apply</button>
        </div>
    );
};
