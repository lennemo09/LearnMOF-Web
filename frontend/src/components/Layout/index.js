import React from 'react';
import MainLayout from './MainLayout';
import Header from "../Header";
import {ToastContainer} from "react-toastify";


export default function Layout() {
  return (
      <div
        className="u-flex u-flexColumn u-flexGrow1 u-overflowHorizontalHidden"
        style={{
          width: 'calc(100%)',
        }}
      >
          <Header />
        <MainLayout />

          <ToastContainer/>
      </div>
  );
}
