import React, { useState } from "react";
import "../styles/Message.css";
import { Document, Page } from "react-pdf";
import Resume from "../files/Ali AlShammaa-Resume.pdf";

export default function ANN() {
  let pageNumber = 1;
  let numPages = 3;
  return (
    <div className="msg container-fluid pt-0 pr-0 pl-0">
      <br />
      <Document file="https://alialshammaa.me/static/media/Ali%20AlShammaa-Resume.3d160214.pdf">
        <Page pageNumber={pageNumber} />
        <p>
          Page {pageNumber} of {numPages}
        </p>
      </Document>
      <hr className="msg-hr" />
    </div>
  );
}
