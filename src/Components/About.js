import React from "react";
import "../styles/About.css";

export default function About() {
  return (
    <div className="about col-12 container-fluid">
      <div className="row ">
        <div className="col-11 col-md-9 col-lg-8 ml-4 mt-4">
          <h1>About me</h1>
          <div className="mt-4 ml-3">
            <p>
              I am a 1st year Computer Science student at UWaterloo in Ontario,
              Canada. I have a passion for Deep learning, ANNs and Intelligence. I love studying
              the (mathematical/CS) theory of ANNs models and enjoy researching new models/ideas to implement AI.
            </p>
            <br />
            <p>
              I recently finished my 1st year term with 91% Cumulative Avg and 92%
              Math Avg. I look forward to learning more CS and Math at UW.
            </p>
            <br />
            <p>
              
            </p>
            <br />
            <p>
              In the future, I see myself researching new technologies from
              Artificial Intelligence to Quantum Computers.
            </p>
            <br />
            <p></p>
          </div>
        </div>

        <div className="col-4">{/* <History /> */}</div>
        {/* <div className="col-4 pl-5 ml-5 mr-0 pr-0"></div> */}
      </div>
    </div>
  );
}
