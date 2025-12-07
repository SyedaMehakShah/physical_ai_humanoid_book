# Plan: Physical AI & Humanoid Robotics – Book Skeleton & Docusaurus Setup

## Architecture Decision Record (ADR)

### ADR1: Docusaurus 3.x + MDX
- **Decision**: Use Docusaurus 3.x with MDX for content
- **Rationale**: Industry standard for tech books; built-in dark mode, SEO, GitHub Pages support; MDX allows React components later
- **Alternatives Considered**: Hugo (simpler), Next.js (more complex), GitBook (proprietary)

### ADR2: GitHub Pages Deployment
- **Decision**: Deploy via GitHub Actions to `gh-pages` branch
- **Rationale**: Free, integrated, no external services needed
- **Build Command**: `npm run build && npm run serve`

### ADR3: Spec-Kit Plus Workflow
- **Decision**: All features follow `/sp.specify` → `/sp.plan` → `/sp.tasks` → `/sp.implement`
- **Rationale**: Ensures consistency, traceability, and quality; prevents ad-hoc changes
- **Implementation**: Add `.specify/` folder with Constitution and specs for each module

### ADR4: Module Structure
- **Decision**: 5 modules total (Quarter Overview + 4 deep modules)
- **Rationale**: Covers ROS 2, Gazebo, Unity, NVIDIA Isaac, and real-world deployment
- **Expansion Plan**: Future modules can be added following same spec-driven workflow

## Technical Implementation Plan

### Phase 1: Docusaurus Setup (2-3 hours)
1. Initialize Docusaurus project: `npx create-docusaurus@latest physical-ai-book classic`
2. Install dependencies: `npm install`
3. Clean up default content (keep structure)
4. Configure `docusaurus.config.js`:
   - Set baseUrl and url for GitHub Pages
   - Configure navbar, footer, sidebar
   - Add social links and favicon
5. Test locally: `npm run start`

### Phase 2: Information Architecture (3-4 hours)
1. Create folder structure in `docs/`:
   - `intro.md` (Quarter Overview)
   - `module-1-ros2/` with 6 sub-pages (index.md, nodes.md, topics.md, services.md, rclpy.md, urdf.md)
   - `module-2-digital-twin/` with 5 sub-pages (index.md, gazebo.md, unity.md, physics.md, sensors.md)
   - `module-3-isaac/` with 3 sub-pages (index.md, isaac-sim.md, ai-integration.md)
   - `module-4-real-robots/` with 4 sub-pages (index.md, robot-hardware.md, deployment.md, safety.md)
   - `module-5-hri-safety/` with 4 sub-pages (index.md, interaction-design.md, safety-protocols.md, ethics.md)
   - `resources/` with tools.md, glossary.md, faq.md
   - `labs/` with lab-template.md and starter files

2. Create basic MDX content for each page:
   - Add frontmatter with title, sidebar_position, description
   - Include placeholder content following pedagogy sequence (Intuition → Concept → Diagram → Example → Exercises)
   - Add learning outcomes for each module

3. Configure sidebar navigation in `sidebars.js`:
   - Group pages by module with collapsible sections
   - Set proper ordering and hierarchy
   - Add future modules as collapsed/grayed items

### Phase 3: Styling & Customization (2-3 hours)
1. Customize theme and styling:
   - Update color scheme to match robotics/AI theme (blues, grays, accent colors)
   - Add custom CSS in `src/css/custom.css`
   - Configure dark/light mode preferences

2. Create custom components:
   - Learning outcome cards component
   - Code example tabs for multiple languages
   - Interactive diagram placeholders
   - Quiz components (for future implementation)

3. Add branding elements:
   - Custom logo/favicon
   - Hero section on homepage
   - Consistent typography and spacing

### Phase 4: Spec-Kit Integration (1-2 hours)
1. Set up `.specify/` folder structure:
   - Create constitution.md with project principles
   - Add template files for future specs
   - Document workflow in README

2. Link content to specifications:
   - Add spec references to each MDX file
   - Create traceability matrix
   - Document how to create new modules following spec-driven approach

### Phase 5: Deployment Setup (1-2 hours)
1. Configure GitHub Actions:
   - Create `.github/workflows/deploy.yml`
   - Set up build and deployment steps
   - Add build status badge to README

2. Configure GitHub Pages:
   - Set source branch to `gh-pages` in repo settings
   - Verify baseUrl configuration works correctly
   - Test deployment with sample commit

3. Add quality checks:
   - Set up linting for MDX files
   - Add build verification steps
   - Create deployment verification process

## Success Criteria

### Technical Verification
- [ ] Docusaurus site builds successfully: `npm run build`
- [ ] Local development works: `npm run start`
- [ ] All navigation links function correctly
- [ ] GitHub Actions workflow deploys to gh-pages
- [ ] Site loads correctly on GitHub Pages
- [ ] Mobile responsiveness verified
- [ ] Accessibility checks pass (WCAG 2.1 AA)

### Content Verification
- [ ] All 5 module folders created with proper structure
- [ ] Sidebar shows correct hierarchy and navigation
- [ ] Learning outcomes present on each module page
- [ ] Spec traceability links present and functional
- [ ] Placeholder content follows pedagogy sequence

### Process Verification
- [ ] Spec-Kit Plus workflow documented and functional
- [ ] New module creation process clearly defined
- [ ] Contribution guidelines updated in README
- [ ] Quality assurance steps documented

## Risk Mitigation

### Technical Risks
- **Docusaurus version compatibility**: Use latest stable version with LTS support
- **GitHub Pages limitations**: Verify all features work within static site constraints
- **Performance**: Optimize images and assets for fast loading

### Content Risks
- **Scope creep**: Focus on skeleton structure, avoid detailed content creation initially
- **Technology changes**: Design modular architecture to accommodate updates to ROS 2, Gazebo, etc.
- **Maintenance**: Establish clear spec-driven process for future updates

## Next Steps

1. Execute Phase 1: Set up Docusaurus foundation
2. Execute Phase 2: Create information architecture
3. Execute Phase 3: Apply styling and customization
4. Execute Phase 4: Integrate Spec-Kit workflow
5. Execute Phase 5: Configure deployment pipeline
6. Verify all success criteria are met
7. Document any deviations from plan and lessons learned

## Estimated Timeline
- Total estimated effort: 9-14 hours across all phases
- Recommended approach: Complete Phase 1-2 in first session, remaining phases in subsequent sessions
- Parallel work possible: Styling can be developed while content structure is being created

**Last Updated**: December 2025
**Owner**: Physical AI Book Team