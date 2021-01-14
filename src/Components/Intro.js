import React, { useState } from "react";
import "../styles/Intro.css";
import picOfMe from "../Images/profilePic1.jpg";
import Resume from "../files/Ali AlShammaa-Resume.pdf";

export default function Intro() {
  useState(false);

  return (
    <div id="intro" className="container-fluid padding intro mt-4">
      <div className="row no-gutters " id="innerintro">
        <div className="col-0 col-md-1"> </div>
        <div className="col-md-4 col-12 pl-5 mr-0 pr-0">
          <div className="">
            <img
              className="picOfMe block-responsive rounded mb-4 silhouette"
              src={picOfMe}
              alt="Ali's profile pic"
            />
          </div>{" "}
        </div>
        {/* <div className="container-fluid">
              <div className="row">
                <i className="col-md-2 fa fa-code fa-4x m-3"></i>
                <i className="col-md-2 fa fa-chrome chrome-grad fa-4x m-3"></i>
                <i className="col-md-2 fa fa-laptop fa-4x m-3"></i>
                <i className="col-md-2 fa fa-book fa-4x m-3"></i>
                <i className="col-md-2 fa fa-university fa-4x m-3"></i>
                <i className="col-md-2 fa fa-fa-sort-numeric-asc fa-4x m-3"></i>
              </div>
            </div> */}
        <div className=" col-md-6  ml-0 pl-0 welcome text-white">
          <hr />
          <h1 className="display-3 text-center">
            <span className="display-4">👋 </span>
            Hi! I'm
            <span className="underline-expand">
              <strong> Ali </strong>
            </span>
            <span className="display-4">😀 </span>
          </h1>
          <hr /> <br />
          <strong>
            <p className="text-center welcome">
              Nice to see you! My name is Ali Al Shammaa
            </p>
          </strong>
          <div className="conatiner-fluid">
            <div className="row">
              <br />
              <div className="col-md-6 col-12">
                <p> I love programming!!</p>
                <p>
                  I am interested in :
                  <br />
                  <ul>
                    <li className="mb-0 mt-0"> Web development</li>
                    <li className="mb-0 mt-0"> Machine learning </li>
                    <li> Quantum Algorithims </li>
                  </ul>
                </p>
              </div>

              <div id="situation" className="col-md-6 col-12 exp mt-">
                <p>
                  {" "}
                  I am currently @<a className="btn-outline-warning">UW CS</a>
                </p>
                <p>
                  {" "}
                  Prev worked @<a className="btn-outline-success">Atlas365</a>
                </p>
                <p>
                  {" "}
                  Interned @<a className="btn-outline-info">Litrans Lab</a>
                </p>
              </div>
            </div>
          </div>
          <div className="conatiner-fluid mt-4">
            <div className="row text-center">
              {/* <div class="btn-group" role="group" aria-label="Basic example"> */}
              <div className="col-sm-6 col-lg-4">
                <span className="git border border-info px-3 py-3 rounded btn-info btn-outline-dark">
                  <a href="https://github.com/AliAlShammaa/">
                    <i className="fa fa-2x fa-github"></i> GitHub{" "}
                  </a>
                </span>
              </div>

              <div className="col-sm-6 col-lg-4">
                <span className=" linkedin border border-info px-3 py-3 mb-1 mx-0 rounded btn-info btn-outline-dark ">
                  <a href="https://www.linkedin.com/in/ali-al-shammaa-508a59bb/">
                    <i className="fa fa-2x fa-linkedin"></i> Linkedin{" "}
                  </a>
                </span>
              </div>
              <br></br>
              <div className="col-lg-4">
                <span className="linkedin border border-info px-3 py-3 mx-0 rounded btn-info btn-outline-dark">
                  {" "}
                  <a href={Resume} target="blank">
                    <i className="fa fa-2x fa-file-pdf-o"></i> Resume/CV{" "}
                  </a>
                </span>
              </div>
              {/* </div> */}
            </div>
          </div>
          <br /> <br /> <br />{" "}
        </div>
      </div>

      <div className="row text-white padding mt-0">
        <div className="col-5"> </div>
        <div className="col-6"></div>
      </div>
    </div>
  );
}
