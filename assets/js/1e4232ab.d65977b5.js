"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[8818],{3905:(e,t,a)=>{a.d(t,{Zo:()=>u,kt:()=>h});var r=a(7294);function n(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function i(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,r)}return a}function o(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?i(Object(a),!0).forEach((function(t){n(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):i(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function l(e,t){if(null==e)return{};var a,r,n=function(e,t){if(null==e)return{};var a,r,n={},i=Object.keys(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||(n[a]=e[a]);return n}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(r=0;r<i.length;r++)a=i[r],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(n[a]=e[a])}return n}var s=r.createContext({}),p=function(e){var t=r.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):o(o({},t),e)),a},u=function(e){var t=p(e.components);return r.createElement(s.Provider,{value:t},e.children)},c="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var a=e.components,n=e.mdxType,i=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),c=p(a),d=n,h=c["".concat(s,".").concat(d)]||c[d]||m[d]||i;return a?r.createElement(h,o(o({ref:t},u),{},{components:a})):r.createElement(h,o({ref:t},u))}));function h(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var i=a.length,o=new Array(i);o[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l[c]="string"==typeof e?e:n,o[1]=l;for(var p=2;p<i;p++)o[p]=a[p];return r.createElement.apply(null,o)}return r.createElement.apply(null,a)}d.displayName="MDXCreateElement"},6193:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>m,frontMatter:()=>i,metadata:()=>l,toc:()=>p});var r=a(7462),n=(a(7294),a(3905));const i={sidebar_position:2},o="Computing the Mean Cell Shape",l={unversionedId:"tutorial-basics/create-a-document",id:"tutorial-basics/create-a-document",title:"Computing the Mean Cell Shape",description:"This guide will walk you through each step in the process, ensuring you get the most out of our tool.",source:"@site/docs/tutorial-basics/create-a-document.md",sourceDirName:"tutorial-basics",slug:"/tutorial-basics/create-a-document",permalink:"/cellgeometry/docs/tutorial-basics/create-a-document",draft:!1,editUrl:"https://github.com/bioshape-lab/cellgeometry/tree/main/packages/create-docusaurus/templates/shared/docs/tutorial-basics/create-a-document.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"\ud83d\udcd8 Formatting your Input Data",permalink:"/cellgeometry/docs/tutorial-basics/create-a-page"},next:{title:"Create a Blog Post",permalink:"/cellgeometry/docs/tutorial-basics/create-a-blog-post"}},s={},p=[{value:"\ud83d\udcdc Step-by-Step Walkthrough",id:"-step-by-step-walkthrough",level:3},{value:"\ud83d\ude80 Usage Tips:",id:"-usage-tips",level:3}],u={toc:p},c="wrapper";function m(e){let{components:t,...a}=e;return(0,n.kt)(c,(0,r.Z)({},u,a,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"computing-the-mean-cell-shape"},"Computing the Mean Cell Shape"),(0,n.kt)("p",null,"This guide will walk you through each step in the process, ensuring you get the most out of our tool."),(0,n.kt)("h3",{id:"-step-by-step-walkthrough"},"\ud83d\udcdc Step-by-Step Walkthrough"),(0,n.kt)("ol",null,(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Initialization and Sidebar Header"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},'The sidebar displays the header "STEP 2: Compute Mean Shape".'))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Data Selection"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"The app first checks if you have uploaded or selected a dataset.",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},'If not, a warning prompts you to upload or select data using the "Load Data" option.'),(0,n.kt)("li",{parentName:"ul"},"If data is already uploaded, the filename is displayed for confirmation."))))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Step Zero - Data Uploading"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},'If you haven\'t uploaded your data, navigate to the "Load Data" page and follow the instructions to ensure proper formatting.'))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Analyzing Cell Data"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"The app provides preprocessing steps, including interpolation, duplicate removal, and quotienting, to prepare your data for analysis."))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Sampling Points Selection"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"Use the slider to select the number of sampling points, which are crucial for the shape analysis. The default is set to 150."))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Data Preprocessing"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"Your data undergoes preprocessing to convert it into a suitable format for analysis."))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Exploring the Geodesic Trajectory")," (Optional):"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},'If you wish to explore the geodesic trajectory between two cell shapes, activate the "Explore Geodesic Trajectory" toggle in the sidebar.'),(0,n.kt)("li",{parentName:"ul"},"Choose the desired treatment and cell line for your analysis."),(0,n.kt)("li",{parentName:"ul"},"Use the provided sliders to select cell indices."),(0,n.kt)("li",{parentName:"ul"},"The geodesic trajectory between the two chosen cell shapes is then visualized in a series of plots."))),(0,n.kt)("li",{parentName:"ol"},(0,n.kt)("p",{parentName:"li"},(0,n.kt)("strong",{parentName:"p"},"Compute Mean Shape"),":"),(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},'Activate the "Compute Mean Shape" toggle in the sidebar to begin this analysis.'),(0,n.kt)("li",{parentName:"ul"},"The computed mean shape will be displayed, which is essentially an average representation of all the cell shapes in your dataset."),(0,n.kt)("li",{parentName:"ul"},"Three plots provide visual insights:",(0,n.kt)("ul",{parentName:"li"},(0,n.kt)("li",{parentName:"ul"},"A plot showcasing the mean estimate."),(0,n.kt)("li",{parentName:"ul"},"A combined plot illustrating individual cell shapes alongside the mean estimate."),(0,n.kt)("li",{parentName:"ul"},"A global mean shape superimposed on the entire dataset of cells.")))))),(0,n.kt)("h3",{id:"-usage-tips"},"\ud83d\ude80 Usage Tips:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"Always Check Data"),": Ensure you've uploaded the correct dataset before proceeding with the analysis."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"Sampling Points"),": Adjusting the number of sampling points can provide different levels of granularity in the analysis. However, higher values may increase computation time."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"Geodesic Trajectory"),": This visualization helps understand the transition between two cell shapes, useful for comparing different treatments or cell lines.")),(0,n.kt)("p",null,"\ud83d\udd0d ",(0,n.kt)("strong",{parentName:"p"},"Note"),": Proper data preparation and understanding of each step are key to extracting meaningful insights from your cell shape data. Happy Analyzing! \ud83c\udf89"))}m.isMDXComponent=!0}}]);