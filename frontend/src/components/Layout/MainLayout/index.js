import React from 'react';
import {Switch, Route, Redirect} from 'react-router-dom';
import FileUpload from "./Upload";
import Browse from './Browse'
import ImageDetail from "./ImageDetail";
import Home from "./Home";
import '../../../App.css';

export default function MainLayout() {
    return (
        <div className='content-box'>
            <div style={{ marginLeft: '20px' }}>
                <Switch>
                    <Route exact path="/" component={Home}/>
                    <Route exact path="/upload" component={FileUpload}/>
                    <Route exact path="/browse" component={Browse}/>
                    <Route path="/browse/:image_url" component={ImageDetail}/>
                    {/* Unknown routes handle */}
                    <Redirect to="/"/>
                </Switch>
            </div>
        </div>
    );
}
